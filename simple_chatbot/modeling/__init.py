"""
The Encoder-Decoder architecture consists of two main components: an Encoder that encodes the input text into a fixed-size /
representation, and a Decoder that decodes the representation to generate the output text.

Here are the steps involved in implementing the Encoder-Decoder architecture:

1- Define the Encoder and Decoder classes:
    We define two classes: an Encoder class and a Decoder class. The Encoder class takes in the input text and produces a /
    fixed-size representation, while the Decoder class takes in the representation and produces the output text.

2- Define the Seq2Seq class:
    We will now use endoer and decoder classes to define the Seq2Seq class. The Seq2Seq class takes in the input text, /
    input_lengths, output text, and teacher_forcing_ratio, and produces the output text. The teacher_forcing_ratio is a /
    hyperparameter that determines whether to use the ground truth output text or the predicted output text from the previous /
    time step as input to the Decoder.
"""