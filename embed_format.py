'''
convert the pre-trained word embedding to .npy format
'''
import gensim
import numpy as np
def convert_google_embedding( 
                            google_embed_file, 
                            new_words_file,
                            new_embed_npfile): 
  model = gensim.models.KeyedVectors.load_word2vec_format(google_embed_file, 
                                                          binary=True)
  print('load finished.')
  
  embed = []
  with open(new_words_file, 'w') as f:
    for i, w in enumerate(model.index2word):
      if i%1000000 == 0:
        print(i)
      embed.append(model[w])
      f.write('%s\n'%w)
  
  embed = np.asarray(embed)
  print('converted to np array')
  embed = embed.astype(np.float32)
  print('converted to float32')
  np.save(new_embed_npfile, embed)


convert_google_embedding('data/GoogleNews-vectors-negative300.bin',
                         'data/google_words.lst',
                         'data/google_embed300.npy')
  
