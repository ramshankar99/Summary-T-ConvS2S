'''
Calculate rouge for one summary-document pair 
'''
import nltk
nltk.download('punkt')

import rouge
import argparse 


def prepare_results(metric, p, r, f):
    '''
    p: precision
    r: recall 
    f: f measure score 
    '''
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

def calculate_rouge(summary, mod_sum):
    for aggregator in ['Best']:
        # print('Evaluation with {}'.format(aggregator))
        apply_avg = aggregator == 'Avg'
        apply_best = aggregator == 'Best'

        evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l'],
                              max_n=2,
                              limit_length=True,
                              length_limit=100,
                              length_limit_type='words',
                              apply_avg=apply_avg,
                              apply_best=apply_best,
                              alpha=0.5, # Default F1_score
                              weight_factor=1.2, # w 
                              stemming=True)
        
        all_hypothesis = []
        all_references = []

        for line in summary.readlines():
            all_hypothesis.append(line) 

        for line in mod_sum.readlines():
            all_references.append(line)  
        print(all_hypothesis)

        if not len(all_hypothesis) == len(all_references):
          print("Check files")
          exit(0)

        for i in range(len(all_hypothesis)):
          scores = evaluator.get_scores(all_hypothesis[i], all_references[i])

          for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            print(prepare_results(metric, results['p'], results['r'], results['f']))
          print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finding rouge results')
    parser.add_argument('--summary', help='Give path to system summary.txt file', default='./rouge/sample_summary.txt')
    parser.add_argument('--mod_sum', help='Give path to model summary.txt file', default='./rouge/gold_summary.txt')
    args = parser.parse_args()

    sum_file_name = args.summary
    doc_file_name = args.mod_sum

    try:
      sum_file = open(sum_file_name, "r")
      doc_file = open(doc_file_name, "r")
      calculate_rouge(sum_file, doc_file)
    except IOError:
      print("Check file inputs")
