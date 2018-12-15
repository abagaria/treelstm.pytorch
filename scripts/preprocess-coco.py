"""
Pre-processing script for MS-COCO captioning data.
"""
from __future__ import print_function
import os
import glob
import json

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def dependency_parse(file_path, cp="", tokenize=True):
    print("Dependency parsing {}\n".format(file_path))
    dir_path = os.path.dirname(file_path)
    file_prefix = os.path.splitext(os.path.basename(file_path))[0]

    # path to file with tokenized input captions.
    tokenized_path = os.path.join(dir_path, file_prefix + ".toks")

    # path to file with parent tree structures.
    parent_path = os.path.join(dir_path, file_prefix + ".parents")

    # path to file with relations between different words in caption (eg conjunction, noun etc)
    relations_path = os.path.join(dir_path, file_prefix + ".rels")

    tokenize_flag = "-tokenize - " if tokenize else ""
    cmd = ("java -cp %s DependencyParse -tokpath %s -parentpath %s -relpath %s %s < %s"
           % (cp, tokenized_path, parent_path, relations_path, tokenize_flag, file_path))

    os.system(cmd)


def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')

def extract_captions_into_text_file(path_to_input_json_file, caption_text_file, image_id_text_file):
    with open(path_to_input_json_file, 'r') as json_file:
        data = json.load(json_file)

    captions, image_ids = [], []
    for annotation in data['annotations']: # type: dict
        captions.append(annotation['caption'])
        image_ids.append(annotation['image_id'])

    with open(caption_text_file, 'w') as text_file:
        for caption in captions:
            caption = caption.lstrip()
            caption = caption.split("\n")[0]
            text_file.write(caption.strip()+"\n")

    with open(image_id_text_file, 'w') as text_file:
        for image_id in image_ids:
            iid = str(image_id)
            zero_padding = '0' * (12 - len(iid)) # Each image name has 12 total digits
            prefix = 'COCO_val2014_' if 'val' in path_to_input_json_file else 'COCO_train2014_'
            text_file.write(prefix + zero_padding + iid + '\n')

def parse(path_to_text_file, cp=''):
    dependency_parse(path_to_text_file, cp=cp, tokenize=True)

if __name__ == '__main__':
    print("=" * 80)
    print("Preprocessing MS-COCO Dataset")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    lib_dir = os.path.join(base_dir, 'lib')

    coco_dir = os.path.join(data_dir, 'coco')
    anno_dir = os.path.join(coco_dir, 'annotations')

    # java classpath for calling Stanford parser
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.1-models.jar')])

    # Create text files for the different captions data files
    extract_captions_into_text_file(os.path.join(anno_dir, 'captions_train2014.json'),
                                    os.path.join(anno_dir, 'captions_train2014.txt'),
                                    os.path.join(anno_dir, 'imageIDs_train2014.txt'))
    extract_captions_into_text_file(os.path.join(anno_dir, 'captions_val2014.json'),
                                    os.path.join(anno_dir, 'captions_val2014.txt'),
                                    os.path.join(anno_dir, 'imageIDs_val2014.txt'))

    # parse sentences
    parse(os.path.join(anno_dir, 'captions_train2014.txt'), cp=classpath)
    parse(os.path.join(anno_dir, 'captions_val2014.txt'), cp=classpath)

    # get vocabulary
    build_vocab(
        glob.glob(os.path.join(coco_dir, '*/*.toks')),
        os.path.join(coco_dir, 'vocab.txt')
    )
    build_vocab(
        glob.glob(os.path.join(coco_dir, '*/*.toks')),
        os.path.join(coco_dir, 'vocab-cased.txt'),
        lowercase=False
    )

