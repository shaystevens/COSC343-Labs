from WordEnc.py import WordEnc
# Create the encoder

my_word_enc = WordEnc(max_words=10000)

# Create the dictionary inside the encoder
my_word_enc.fit(some_string_with_text)

# Convert text to one−hot encoded matrix
X = my_word_enc.transform(some_string_with_text)

# Convert one−hot encoded matrix back to a string
a_string = my_word_enc.inverse_transform(X)
