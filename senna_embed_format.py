'''
convert the pre-trained word embedding to .npy format
'''

import numpy as np

def convert_senna_embedding(senna_words_file, 
                            senna_embed_file, 
                            new_words_file,
                            new_embed_npfile,
                            word_dim):
  vocab = []
  # load senna vocab
  with open(senna_words_file) as f:
    for line in f:
      w = line.strip()
      vocab.append(w)
  
  # load senna embedding
  embed = []
  with open(senna_embed_file) as f:
    for line in f:
      vec = [float(x) for x in line.strip().split()]
      assert len(vec) == word_dim
      embed.append(vec)
  
  # write new words file
  with open(new_words_file, 'w') as f:
    for w in vocab:
      f.write('%s\n' % w)
  
  # save embed as .npy format
  embed = np.asarray(embed)
  embed = embed.astype(np.float32)
  np.save(new_embed_npfile, embed)

convert_senna_embedding('data/embedding/senna/words.lst',
                        'data/embedding/senna/embeddings.txt',
                        'data/senna_words.lst',
                        'data/senna_embed50.npy',
                        50)


