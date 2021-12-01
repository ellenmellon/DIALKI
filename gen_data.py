import argparse
from data_utils import Doc2DialReader
from transformers import AutoTokenizer
from config import TOKENS

def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_dir)
    tokenizer.add_special_tokens(
        {
            'additional_special_tokens': TOKENS,
        }
    )

    reader = Doc2DialReader(
        args,
        args.input_dir,
        args.output_dir,
        tokenizer, 
        args.max_seq_len,
        args.max_history_len,
        args.max_num_spans_per_passage,
        args.num_sample_per_file,
    )

    reader.convert_json_to_finetune_pkl('train')
    reader.convert_json_to_finetune_pkl('dev')

    if args.data_name == 'wow':
        reader.convert_json_to_finetune_pkl('test')
        reader.convert_json_to_finetune_pkl('dev_unseen')
        reader.convert_json_to_finetune_pkl('test_unseen')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        required=True,
        type=str,
        choices=['dialdoc', 'wow'],
        help='',
    )
    parser.add_argument(
        '--pretrained_model_dir',
        required=True,
        type=str,
        help='',
    )
    parser.add_argument(
        '--input_dir',
        required=True,
        type=str,
        help='',
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        type=str,
        help='',
    )
    parser.add_argument(
        '--max_seq_len',
        required=True,
        type=int,
        help='',
    )
    parser.add_argument(
        '--max_history_len',
        required=True,
        type=int,
        help='',
    )
    parser.add_argument(
        '--max_num_spans_per_passage',
        required=True,
        type=int,
        help='',
    )
    parser.add_argument(
        '--num_sample_per_file',
        default=1000,
        type=int,
        help='',
    )
    parser.add_argument(
        '--use_cls_span_start',
        action='store_true',
        help='',
    )

    args = parser.parse_args()

    main(args)
