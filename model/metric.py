from nltk.translate.bleu_score import corpus_bleu
'''
calculat_bleu_scores( ): used to calculate the BLeU score of the prediction againts the MS COCO captions 
- all five captions were used as references 
- bleu 1 to bleu 4 were tracked and reported
'''


def calculate_bleu_scores(hypothesis, img_id, vocab, data_sample, weights=(0.25, 0.25, 0.25, 0.25)):

    def tokens_to_word(tokens):
        return [
            vocab.idx2word[idx]
            for idx in tokens
            if idx not in {
                vocab.word2idx['<pad>'],
                vocab.word2idx['<start>'],
                vocab.word2idx['<end>']
            }
        ]

    references = []
    for ids in img_id:
        for entry in data_sample:
            if entry['id'] == ids:
                cleaned_captions = [tokens_to_word(caption) for caption in entry['captions']]
                references.append(cleaned_captions)
                break  

    cleaned_hypotheses = [tokens_to_word(pred) for pred in hypothesis]

    return corpus_bleu(references, cleaned_hypotheses, weights=weights)
