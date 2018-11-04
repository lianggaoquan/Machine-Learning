## basic preprocess of text data

- **Read primitive file** 
- **create lookup tables**:   `vocab_to_int`	& `int_to_vocab`
- **explore the data**:
  - `english_sentences = source_text.split ('\n')`
    `french_sentences = target_text.split ('\n')`
  - `side_by_side_sentences = list(zip(english_sentences,french_sentences)) [0,5]`
  - `for index,sentence in enumerate(side_by_side_sentences):`
    ​    `en_sent,fr_sent = sentence`
    ​    `print(index+1,' sentence:')`
    ​    `print('\tEN:{}'.format(en_sent))`
    ​    `print('\tFR:{}'.format(fr_sent))`
- **text to ids**:
  - `text_to_ids (source_text,  target_text,  source_vocab_to_int,  target_vocab_to_int)`
- **merge all operations above**
- 