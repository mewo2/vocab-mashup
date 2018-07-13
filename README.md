# Vocabulary Mashup

A cheating pseudo-entry in [NaNoGenMo 2015](https://github.com/dariusk/NaNoGenMo-2015/).

This code mashes up two source texts. The first text is used for structure,
while the second provides the vocabulary. Word replacements are chosen to be
semantically close (using word2vec and part-of-speech identification) as well
as similar in frequency between the texts.

pip3 install click, gensim         
python3 mashup.py mash input/alices.txt input/bible.txt        




**Gender mashup
python mashup.py gender input/alices.txt       
Computing frequencies for input/alices.txt       
Loading POS tags       
Loaded       
guinea [('m', 0.0014285714285714286), ('f', 0.0006184291898577613), ('n', 0.0022189349112426036)]       
Cheshire [('m', 0.0), ('f', 0.0018552875695732839), ('n', 0.0029585798816568047)]       
William [('m', 0.005714285714285714), ('f', 0.0012368583797155227), ('n', 0.0014792899408284023)]       
whiting [('m', 0.004285714285714286), ('f', 0.0006184291898577613), ('n', 0.0014792899408284023)]       
brown [('m', 0.0014285714285714286), ('f', 0.0006184291898577613), ('n', 0.005177514792899409)]       
cook [('m', 0.005714285714285714), ('f', 0.0055658627087198514), ('n', 0.0007396449704142012)]       
lily [('m', 0.0), ('f', 0.004947433518862091), ('n', 0.005177514792899409)]       
Kitty [('m', 0.0014285714285714286), ('f', 0.004329004329004329), ('n', 0.011834319526627219)]       
March [('m', 0.01), ('f', 0.0024737167594310453), ('n', 0.013313609467455622)]       
Humpty [('m', 0.017142857142857144), ('f', 0.0074211502782931356), ('n', 0.016272189349112426)]       
Dumpty [('m', 0.017142857142857144), ('f', 0.0074211502782931356), ('n', 0.016272189349112426)]       
Alice [('m', 0.14), ('f', 0.22077922077922077), ('n', 0.1952662721893491)]       