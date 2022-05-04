import argparse
from argparse import Namespace
from fairseq.data import encoders
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input-file", type=str, required=True)
parser.add_argument("--output-file", type=str, required=True)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--add_lang_tag", type=str, default=None)

args = parser.parse_args()

bpe_tokenizer = encoders.build_bpe(
    Namespace(
        bpe='sentencepiece',
        sentencepiece_model=args.model,
    )
)

with open(args.input_file, 'rb') as input_file:
    with open(args.output_file, 'w') as output_file:
        for line in tqdm.tqdm(input_file):
            line = str(line.decode("utf-8")).strip()
            encoded_line = bpe_tokenizer.encode(line).strip()
            if args.add_lang_tag is None:
                output_file.write(encoded_line + '\n')
            else:
                output_file.write(f"<lang:{args.add_lang_tag}> " + encoded_line + "\n")