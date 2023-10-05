from json_utils import save_json_l, load_json, load_json_l_qa


video_to_transcript_dict = load_json('video_to_transcripts.json')

qa_train = load_json_l_qa('../data/qa/qa_train.json')
new_qa_train = []
for i, qa in qa_train.items():
    if qa['vid_name'] in video_to_transcript_dict.keys():
        new_qa_train.append(qa)
save_json_l('../data/qa/qa_train_processed.jsonl', new_qa_train)

qa_val = load_json_l_qa('../data/qa/qa_val.json')
new_qa_val = []
for i, qa in qa_val.items():
    if qa['vid_name'] in video_to_transcript_dict.keys():
        new_qa_val.append(qa)
save_json_l('../data/qa/qa_val_processed.jsonl', new_qa_val)
#
qa_test = load_json_l_qa('../data/qa/qa_test.json')
new_qa_test = []
for i, qa in qa_test.items():
    if qa['vid_name'] in video_to_transcript_dict.keys():
        new_qa_test.append(qa)
save_json_l('../data/qa/qa_test_processed.jsonl', new_qa_test)
