import sentencepiece as spm
import constant

def build_sp(is_train=True, filenames=None, fileiter=None):
    if is_train:
        if type(filenames) is list:
            assert len(filenames) == len(fileiter)
            concat_files(filenames, fileiter, constant.SP_path)
            input_file = constant.SP_path
        else:
            input_file = filenames
            
        templates = '--input={} --model_prefix={} --vocab_size={}'
        cmd = templates.format(input_file, constant.SP_prefix, constant.vocab_size)
        spm.SentencePieceTrainer.Train(cmd)
    sp = spm.SentencePieceProcessor()
    sp.Load('{}.model'.format(constant.SP_prefix))
    with open('{}.vocab'.format(constant.SP_prefix), encoding='utf-8') as f:
        vocabs = [doc.strip().split('\t')[0] for doc in f]
    return sp, vocabs

def concat_files(filenames, fileiter, out_filename):
    with open(out_filename, 'w') as outfile:
        for i, fname in enumerate(filenames):
            for iteration in range(fileiter[i]):
                with open(fname) as infile:
                    for line in infile:
                        outfile.write(line)