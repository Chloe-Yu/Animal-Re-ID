import sys
import stat
import fnmatch
import os
if os.name == 'posix':
    import posix
  # macOS
def _stat(fn):
    return fn.stat() if isinstance(fn, os.DirEntry) else os.stat(fn)

COPY_BUFSIZE = 1024 * 1024
_WINDOWS = os.name == 'nt'
_USE_CP_SENDFILE = hasattr(os, "sendfile") and sys.platform.startswith("linux")
def _fastcopy_fcopyfile(fsrc, fdst, flags):
    """Copy a regular file content or metadata by using high-performance
    fcopyfile(3) syscall (macOS).
    """
    try:
        infd = fsrc.fileno()
        outfd = fdst.fileno()
    except Exception as err:
        raise _GiveupOnFastCopy(err)  # not a regular file

    try:
        posix._fcopyfile(infd, outfd, flags)
    except OSError as err:
        err.filename = fsrc.name
        err.filename2 = fdst.name
        if err.errno in {errno.EINVAL, errno.ENOTSUP}:
            raise _GiveupOnFastCopy(err)
        else:
            raise err from None

def _fastcopy_sendfile(fsrc, fdst):
    """Copy data from one regular mmap-like fd to another by using
    high-performance sendfile(2) syscall.
    This should work on Linux >= 2.6.33 only.
    """
    # Note: copyfileobj() is left alone in order to not introduce any
    # unexpected breakage. Possible risks by using zero-copy calls
    # in copyfileobj() are:
    # - fdst cannot be open in "a"(ppend) mode
    # - fsrc and fdst may be open in "t"(ext) mode
    # - fsrc may be a BufferedReader (which hides unread data in a buffer),
    #   GzipFile (which decompresses data), HTTPResponse (which decodes
    #   chunks).
    # - possibly others (e.g. encrypted fs/partition?)
    global _USE_CP_SENDFILE
    try:
        infd = fsrc.fileno()
        outfd = fdst.fileno()
    except Exception as err:
        raise _GiveupOnFastCopy(err)  # not a regular file

    # Hopefully the whole file will be copied in a single call.
    # sendfile() is called in a loop 'till EOF is reached (0 return)
    # so a bufsize smaller or bigger than the actual file size
    # should not make any difference, also in case the file content
    # changes while being copied.
    try:
        blocksize = max(os.fstat(infd).st_size, 2 ** 23)  # min 8MiB
    except OSError:
        blocksize = 2 ** 27  # 128MiB
    # On 32-bit architectures truncate to 1GiB to avoid OverflowError,
    # see bpo-38319.
    if sys.maxsize < 2 ** 32:
        blocksize = min(blocksize, 2 ** 30)

    offset = 0
    while True:
        try:
            sent = os.sendfile(outfd, infd, offset, blocksize)
        except OSError as err:
            # ...in oder to have a more informative exception.
            err.filename = fsrc.name
            err.filename2 = fdst.name

            if err.errno == errno.ENOTSOCK:
                # sendfile() on this platform (probably Linux < 2.6.33)
                # does not support copies between regular files (only
                # sockets).
                _USE_CP_SENDFILE = False
                raise _GiveupOnFastCopy(err)

            if err.errno == errno.ENOSPC:  # filesystem is full
                raise err from None

            # Give up on first call and if no data was copied.
            if offset == 0 and os.lseek(outfd, 0, os.SEEK_CUR) == 0:
                raise _GiveupOnFastCopy(err)

            raise err
        else:
            if sent == 0:
                break  # EOF
            offset += sent

def _copyfileobj_readinto(fsrc, fdst, length=COPY_BUFSIZE):
    """readinto()/memoryview() based variant of copyfileobj().
    *fsrc* must support readinto() method and both files must be
    open in binary mode.
    """
    # Localize variable access to minimize overhead.
    fsrc_readinto = fsrc.readinto
    fdst_write = fdst.write
    with memoryview(bytearray(length)) as mv:
        while True:
            n = fsrc_readinto(mv)
            if not n:
                break
            elif n < length:
                with mv[:n] as smv:
                    fdst.write(smv)
            else:
                fdst_write(mv)

def copyfileobj(fsrc, fdst, length=0):
    """copy data from file-like object fsrc to file-like object fdst"""
    # Localize variable access to minimize overhead.
    if not length:
        length = COPY_BUFSIZE
    fsrc_read = fsrc.read
    fdst_write = fdst.write
    while True:
        buf = fsrc_read(length)
        if not buf:
            break
        fdst_write(buf)


def _samefile(src, dst):
    # Macintosh, Unix.
    if isinstance(src, os.DirEntry) and hasattr(os.path, 'samestat'):
        try:
            return os.path.samestat(src.stat(), os.stat(dst))
        except OSError:
            return False

    if hasattr(os.path, 'samefile'):
        try:
            return os.path.samefile(src, dst)
        except OSError:
            return False

    # All other platforms: check for same pathname.
    return (os.path.normcase(os.path.abspath(src)) ==
            os.path.normcase(os.path.abspath(dst)))


def copyfile(src, dst, *, follow_symlinks=True):
    _HAS_FCOPYFILE = posix and hasattr(posix, "_fcopyfile")
    """Copy data from src to dst in the most efficient way possible.

    If follow_symlinks is not set and src is a symbolic link, a new
    symlink will be created instead of copying the file it points to.

    """
    sys.audit("shutil.copyfile", src, dst)

    if _samefile(src, dst):
        raise SameFileError("{!r} and {!r} are the same file".format(src, dst))

    file_size = 0
    for i, fn in enumerate([src, dst]):
        try:
            st = _stat(fn)
        except OSError:
            # File most likely does not exist
            pass
        else:
            # XXX What about other special files? (sockets, devices...)
            if stat.S_ISFIFO(st.st_mode):
                fn = fn.path if isinstance(fn, os.DirEntry) else fn
                raise SpecialFileError("`%s` is a named pipe" % fn)
            if _WINDOWS and i == 0:
                file_size = st.st_size

    if not follow_symlinks and _islink(src):
        os.symlink(os.readlink(src), dst)
    else:
        with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
            # macOS
            if _HAS_FCOPYFILE:
                try:
                    _fastcopy_fcopyfile(fsrc, fdst, posix._COPYFILE_DATA)
                    return dst
                except _GiveupOnFastCopy:
                    pass
            # Linux
            elif _USE_CP_SENDFILE:
                try:
                    _fastcopy_sendfile(fsrc, fdst)
                    return dst
                except _GiveupOnFastCopy:
                    pass
            # Windows, see:
            # https://github.com/python/cpython/pull/7160#discussion_r195405230
            elif _WINDOWS and file_size > 0:
                _copyfileobj_readinto(fsrc, fdst, min(file_size, COPY_BUFSIZE))
                return dst

            copyfileobj(fsrc, fdst)

    return dst
