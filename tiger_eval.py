# from https://github.com/cvwc2019/ATRWEvalScript
import argparse
import random
import os,sys
import numpy as np
import json
from sklearn.metrics import average_precision_score

#sys.path.insert(0,os.path.join(os.path.dirname(__file__), '../'))


def eval_impl(anno,res,path,partial=False):
    random.seed(114514)
    with open(anno,'r') as f:
        anno=json.load(f)
    if path:
        with open(res,'r') as f:
            res=json.load(f)
    eids,fids={},{}
    query_multi=[]
    query_sing=[]
    for obj in anno:
        imgid,query,frame,entityid=obj['imgid'],obj['query'],obj['frame'],obj['entityid']
        if query=='multi':
            query_multi.append(imgid)
        elif query=='sing':
            query_sing.append(imgid)
        eids[imgid]=entityid
        fids[imgid]=tuple(frame)
    #gt loaded
    # DTODO: split partial
    if partial:
        random.shuffle(query_sing)
        random.shuffle(query_multi)
        query_sing=query_sing[:len(query_sing)//2]
        query_multi=query_multi[:len(query_sing)//2]


    aps_sing,aps_multi=[],[]
    cmc_sing,cmc_multi = np.zeros(len(eids), dtype=np.int32),np.zeros(len(eids), dtype=np.int32)
    for ans in res:
        queryid,ans_ids=ans['query_id'],ans['ans_ids']
        if queryid in query_multi:
            aps=aps_multi
        elif queryid in query_sing:
            aps=aps_sing
        else:
            continue
        entityid=eids[queryid]
        gt_eids=np.asarray([eids[i] for i in ans_ids])
        pid_matches= gt_eids==entityid
        mask=exclude(entityid,fids[queryid],gt_eids,[fids[i] for i in ans_ids])
        distances=np.arange(len(ans_ids))+1.0
        distances[mask] = np.inf
        pid_matches[mask] = False
        scores = 1 / distances
        ap = average_precision_score(pid_matches, scores)
        if np.isnan(ap):
            print()
            print("WARNING: encountered an AP of NaN!")
            print("This usually means a person only appears once.")
            print("In this case, it's because of {}.".format(fids[i]))
            print("I'm excluding this person from eval and carrying on.")
            print()
            aps.append(0)
            continue
        aps.append(ap)
        k = np.where(pid_matches[np.argsort(distances)])[0][0]
        if queryid in query_multi:
            cmc_multi[k:] += 1
        else:
            cmc_sing[k:]  += 1
    mean_ap_sing = np.mean(aps_sing)
    mean_ap_multi = np.mean(aps_multi)
    top1_sing=cmc_sing[0]/len(query_sing)
    top5_sing=cmc_sing[4]/len(query_sing)
    top1_multi=cmc_multi[0]/len(query_multi)
    top5_multi=cmc_multi[4]/len(query_multi)
    return {'mmAP':(mean_ap_sing+mean_ap_multi)/2,
            'mAP(single_cam)':mean_ap_sing,
            'top-1(single_cam)':top1_sing,
            'top-5(single_cam)':top5_sing,
            'mAP(cross_cam)':mean_ap_multi,
            'top-1(cross_cam)':top1_multi,
            'top-5(cross_cam)':top5_multi}


def exclude(eid,fid,eids,fids):
    eids,fids=np.asarray(eids),np.asarray(fids)
    eid_match= (eids==eid)
    cid_match= np.logical_and( fid[0]==fids[:,0],
                               fid[1]==fids[:,1])
    frame_match=np.abs(fids[:,2]-fid[2])<=3
    mask = np.logical_and(cid_match, eid_match)
    mask = np.logical_and(mask, frame_match)
    junk_images= eids==-1
    mask = np.logical_or(mask, junk_images)

    return mask

def plain_evaluate(anno, res, phase_codename,path, **kwargs):
    """
    Evaluates the submission for a particular challenge phase adn returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            'status': u'running',
            'when_made_public': None,
            'participant_team': 5,
            'input_file': 'https://abc.xyz/path/to/submission/file.json',
            execution_time': u'123',
            'publication_url': u'ABC',
            'challenge_phase': 1,
            'created_by': u'ABC',
            'stdout_file': 'https://abc.xyz/path/to/stdout/file.json',
            'method_name': u'Test',
            'stderr_file': 'https://abc.xyz/path/to/stderr/file.json',
            'participant_team_name': u'Test Team',
            'project_url': u'http://foo.bar',
            'method_description': u'ABC',
            'is_public': False,
            'submission_result_file': 'https://abc.xyz/path/result/file.json',
            'id': 123,
            'submitted_at': u'2017-03-20T19:22:03.880652Z'
        }
    """

    print("Starting Evaluation.....")
    # print("Submission related metadata:")
    # print(kwargs['submission_metadata'])

    output = {}
    if phase_codename == "dev":
        print("Evaluating for Dev Phase")

        output['result'] = [
            {
                'public_split': eval_impl(anno, res,path,partial=True),
            },
        ]
        print("Completed evaluation for Dev Phase")
    elif phase_codename == "test":
        print("Evaluating for Test Phase")
        output['result'] = [
            {
                'public_split': eval_impl(anno, res,path,partial=True)
            },
            {
                'private_split': eval_impl(anno, res,path),
            }
        ]
        print("Completed evaluation for Test Phase")
    return output



# model={'detect':detect,'pose':pose,'plain':plain,'wild':wild}
annos={
    'detect':'annotations/detect_tiger02_test.json',
    'pose':'annotations/pose_tiger02_test.json',
    'plain':'./data/tiger/atrw_anno_reid_test/gt_test_plain.json',
    'wild':'annotations/gt_test_wild.json'}

def evaluate_tiger(input_file_path, task, path=True, **kwargs):
    output=plain_evaluate(annos[task],input_file_path,'test',path)
    #output["submission_result"] = output["result"][0]["public_split"]
    return output

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    args=parser.parse_args()
    output=evaluate_tiger(args.input,'plain')
    print(output)
