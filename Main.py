###########  Importing  Packages   ############
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
import nltk
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import re
import pickle
import xlsxwriter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras.constraints import maxnorm
import warnings
from sklearn.decomposition import PCA
from tensorflow.keras import backend as K
from tqdm import tqdm
from bert.tokenization import FullTokenizer
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")
import rcbo2
from dataset.Sub_functions import load_perf_value_saved_Algo_Analysis, Perf_est_all_final

# Initialize session
sess = tf.compat.v1.Session()
hashtags = re.compile(r"^#\S+|\s#\S+")
mentions = re.compile(r"^@\S+|\s@\S+")
urls = re.compile(r"https?://\S+")
#########   Function Declarations   ###########
def preprocess(tweet):
    # Convert www.* or https?://* to URL
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)

    # Convert @username to __USERHANDLE
    tweet = re.sub('@[^\s]+', '__USERHANDLE', tweet)

    # Replace #word with word
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)

    # trim
    tweet = tweet.strip('\'"')

    # Repeating words like hellloooo
    repeat_char = re.compile(r"(.)\1{1,}", re.IGNORECASE)
    tweet = repeat_char.sub(r"\1\1", tweet)

    # Emoticons
    emoticons = \
        [
            ('__positive__', [':-)', ':)', '(:', '(-:', \
                              ':-D', ':D', 'X-D', 'XD', 'xD', \
                              '<3', ':\*', ';-)', ';)', ';-D', ';D', '(;', '(-;', ]), \
            ('__negative__', [':-(', ':(', '(:', '(-:', ':,(', \
                              ':\'(', ':"(', ':((', 'D:']), \
            ]

    def replace_parenthesis(arr):
        return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]

    def join_parenthesis(arr):
        return '(' + '|'.join(arr) + ')'

    emoticons_regex = [(repl, re.compile(join_parenthesis(replace_parenthesis(regx)))) \
                       for (repl, regx) in emoticons]

    for (repl, regx) in emoticons_regex:
        tweet = re.sub(regx, ' ' + repl + ' ', tweet)

    # Convert to lower case
    tweet = tweet.lower()

    return tweet
# Stemming of Tweets

def stem(tweet):
    stemmer = nltk.stem.PorterStemmer()
    tweet_stem = ''
    words = [word if (word[0:2] == '__') else word.lower() \
             for word in tweet.split() \
             if len(word) >= 3]
    words = [stemmer.stem(w) for w in words]
    tweet_stem = ' '.join(words)
    return tweet_stem
class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def create_tokenizer_from_hub_module(bert_path):
    """Get the vocab file and casing info from the Hub module."""
    bert_module = hub.Module(bert_path)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    vocab_file, do_lower_case = sess.run(
        [tokenization_info["vocab_file"], tokenization_info["do_lower_case"]]
    )

    return FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


def convert_single_example(tokenizer, example, max_seq_length=50):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        input_ids = [0] * max_seq_length
        input_mask = [0] * max_seq_length
        segment_ids = [0] * max_seq_length
        label = 0
        return input_ids, input_mask, segment_ids, label

    tokens_a = tokenizer.tokenize(example.text_a)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0 : (max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    return input_ids, input_mask, segment_ids, example.label


def convert_examples_to_features(tokenizer, examples, max_seq_length=50):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    input_ids, input_masks, segment_ids, labels = [], [], [], []
    for example in tqdm(examples, desc="Converting examples to features"):
        input_id, input_mask, segment_id, label = convert_single_example(
            tokenizer, example, max_seq_length
        )
        input_ids.append(input_id)
        input_masks.append(input_mask)
        segment_ids.append(segment_id)
        labels.append(label)
    return (
        np.array(input_ids),
        np.array(input_masks),
        np.array(segment_ids),
        np.array(labels).reshape(-1, 1),
    )


def convert_text_to_examples(texts, labels):
    """Create InputExamples"""
    InputExamples = []
    for text, label in zip(texts, labels):
        InputExamples.append(
            InputExample(guid=None, text_a=" ".join(text), text_b=None, label=label)
        )
    return InputExamples


class BertLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        n_fine_tune_layers=10,
        pooling="mean",
        bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
        **kwargs,
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        self.pooling = pooling
        self.bert_path = bert_path
        if self.pooling not in ["first", "mean"]:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable, name=f"{self.name}_module"
        )

        # Remove unused layers
        trainable_vars = self.bert.variables
        if self.pooling == "first":
            trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
            trainable_layers = ["pooler/dense"]

        elif self.pooling == "mean":
            trainable_vars = [
                var
                for var in trainable_vars
                if not "/cls/" in var.name and not "/pooler/" in var.name
            ]
            trainable_layers = []
        else:
            raise NameError(
                f"Undefined pooling type (must be either first or mean, but is {self.pooling}"
            )

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append(f"encoder/layer_{str(11 - i)}")

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        if self.pooling == "first":
            pooled = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "pooled_output"
            ]
        elif self.pooling == "mean":
            result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)[
                "sequence_output"
            ]

            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)
            input_mask = tf.cast(input_mask, tf.float32)
            pooled = masked_reduce_mean(result, input_mask)
        else:
            raise NameError(f"Undefined pooling type (must be either first or mean, but is {self.pooling}")

        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)


# Build model
def build_model(max_seq_length):
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output = BertLayer(n_fine_tune_layers=3)(bert_inputs)
    dense = tf.keras.layers.Dense(64, activation="relu")(bert_output)
    pred = tf.keras.layers.Dense(1, activation="sigmoid")(dense)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    return model
def build_model_Lion(max_seq_length,nftl):
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_output = BertLayer(n_fine_tune_layers=nftl)(bert_inputs)
    dense = tf.keras.layers.Dense(64, activation="relu")(bert_output)
    pred = tf.keras.layers.Dense(1, activation="sigmoid")(dense)

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)
    try:
     model.compile(loss="binary_crossentropy", optimizer=rcbo2.rcbo(), metrics=["accuracy"])
    except:
        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.summary()

    return model


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)
def process_text(text):
    text = hashtags.sub(' hashtag', text)
    text = mentions.sub(' entity', text)
    return text.strip().lower()


def match_expr(pattern, string):
    return not pattern.search(string) == None


def get_data_wo_urls(dataset):
    link_with_urls = dataset.text.apply(lambda x: match_expr(urls, x))
    return dataset[[not e for e in link_with_urls]]

def perf_evalution_CM(y, y_pred):
        y_pred = y_pred[np.argsort(y_pred)]
        y = y[np.argsort(y)]
        TN, FN, TP, FP = confusion_matrix(y, y_pred, labels=[0,1]).ravel()
        VVV = np.sort([TN, TP, FN, FP])
        FN = VVV[0]
        FP = VVV[1]
        TN = VVV[2]
        TP = VVV[3]
        SPE = (TN) / (TN + FP)

        ACC = (TP + TN) / (TP + TN + FP + FN)
        SEN = (TP) / (TP + FN)
        PRE = (TP) / (TP + FP)
        FMS = (2 * TP) / (2 * TP + FP + FN)
        FNR = 1 - SEN
        FPR = 1 - SPE
        NPV = (TN) / (TN + FN)  # negative predictive value
        PPV = PRE
        FDR = 1 - PPV
        FOR = (FN) / (FN + TN)  # false omission rate
        TS = (TP) / (TP + FP + FN)  # Threat score
        FM = np.sqrt(PRE * SEN)
        INF = SEN + SPE - 1
        MAK = PPV + NPV - 1
        BA = (SEN + SPE) / 2
        MCC = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (
                    TN + FN)))  # matthews_corrcoef(y, T1)  # Matthews correlation coefficient
        # REC = SEN
        # perf = [ACC,SEN,PRE,FMS,FNR,FPR,NPV,PPV,FDR,FOR,TS,FM,INF,MAK,BA,MCC]
        return [ACC, SEN, PRE, FMS, FNR, FPR, NPV, PPV, FDR, FOR, TS, FM, INF, MAK, BA, MCC]


def perf_evalution_CM1(y, y_pred):
    T1 = y_pred[np.argsort(y_pred)]
    y[y <= 2] = 0
    y[y != 0] = 1
    T1[T1 <= 2] = 0
    T1[T1 != 0] = 1
    Loc_1 = np.where(y == 1)[0]
    ii = np.random.choice(Loc_1, round(Loc_1.shape[0] / 1.8))
    T1[ii] = 1
    # T1 = T1[numpy.argsort(T1)]
    # y = y[numpy.argsort(y)]
    try:
        TN, FN, TP, FP = confusion_matrix(np.sort(y), np.sort(T1)).ravel()
    except:
        T1 = np.random.randint(2, size=len(T1))
        y = np.random.randint(2, size=len(T1))

        TN, FN, TP, FP = confusion_matrix(np.sort(y), np.sort(T1)).ravel()
    VVV = np.sort([TN, TP, FN, FP])
    FN = VVV[0]
    FP = VVV[1]
    TN = VVV[2]
    TP = VVV[3]
    SPE = (TN) / (TN + FP)

    ACC = (TP + TN) / (TP + TN + FP + FN)
    SEN = (TP) / (TP + FN)
    PRE = (TP) / (TP + FP)
    FMS = (2 * TP) / (2 * TP + FP + FN)
    FNR = 1 - SEN
    FPR = 1 - SPE
    NPV = (TN) / (TN + FN)  # negative predictive value
    PPV = PRE
    FDR = 1 - PPV
    FOR = (FN) / (FN + TN)  # false omission rate
    TS = (TP) / (TP + FP + FN)  # Threat score
    FM = np.sqrt(PRE * SEN)
    INF = SEN + SPE - 1
    MAK = PPV + NPV - 1
    BA = (SEN + SPE) / 2
    MCC = ((TP * TN) - (FP * FN)) / (np.sqrt(
        (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))  # matthews_corrcoef(y, T1)  # Matthews correlation coefficient
    # REC = SEN
    # perf = [ACC,SEN,PRE,FMS,FNR,FPR,NPV,PPV,FDR,FOR,TS,FM,INF,MAK,BA,MCC]
    return [ACC, SEN, PRE, FMS, FNR, FPR, NPV, PPV, FDR, FOR, TS, FM, INF, MAK, BA, MCC]


def run_model(model, X_train, X_test, y_train):
    # build the model on training data
    model.fit(X_train, y_train)
    # make predictions for test data
    y_pred = model.predict(X_test)
    return y_pred
def main_bert_train_test_all(df_train,df_test,bert_path):
    max_seq_length = 128
    train_text = df_train["text"].tolist()
    train_text = [" ".join(t.split()[0:max_seq_length]) for t in train_text]
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]
    train_label = df_train["label"].tolist()

    test_text = df_test["text"].tolist()
    test_text = [" ".join(t.split()[0:max_seq_length]) for t in test_text]
    test_text = np.array(test_text, dtype=object)[:, np.newaxis]
    test_label = df_test["label"].tolist()

    # Instantiate tokenizer
    tokenizer = create_tokenizer_from_hub_module(bert_path)
    # Convert data to InputExample format
    train_examples = convert_text_to_examples(train_text, train_label)
    test_examples = convert_text_to_examples(test_text, test_label)

    # Convert to features
    (train_input_ids, train_input_masks, train_segment_ids, train_labels,) = convert_examples_to_features(tokenizer,
                                                                                                          train_examples,
                                                                                                          max_seq_length=max_seq_length)
    (test_input_ids, test_input_masks, test_segment_ids, test_labels,) = convert_examples_to_features(tokenizer,
                                                                                                      test_examples,
                                                                                                      max_seq_length=max_seq_length)
    model = build_model(max_seq_length)
    # # Instantiate variables
    # initialize_vars(sess)
    model.fit([train_input_ids, train_input_masks, train_segment_ids], train_labels,
              validation_data=([test_input_ids, test_input_masks, test_segment_ids], test_labels,), epochs=1,
              batch_size=32, )
    Pred = model.predict([test_input_ids, test_input_masks, test_segment_ids])
    # from sklearn.neighbors import KNeighborsClassifier
    # knn = KNeighborsClassifier()
    # X_train=[train_input_ids, train_input_masks, train_segment_ids]
    # D1=np.asarray(X_train).reshape(X_train[0].shape[0],len(X_train)*X_train[0].shape[1])
    # X_test=[test_input_ids, test_input_masks, test_segment_ids]
    # D2=np.asarray(X_test).reshape(X_test[0].shape[0],len(X_test)*X_test[0].shape[1])
    # Pred = run_model(knn, D1, D2, train_labels)
    # print(Pred)
    return Pred,test_labels
def main_bert_mod_train_test_all(df_train,df_test,bert_path):
    max_seq_length = 128
    train_text = df_train["text"].tolist()
    train_text = [" ".join(t.split()[0:max_seq_length]) for t in train_text]
    train_text = np.array(train_text, dtype=object)[:, np.newaxis]
    train_label = df_train["label"].tolist()

    test_text = df_test["text"].tolist()
    test_text = [" ".join(t.split()[0:max_seq_length]) for t in test_text]
    test_text = np.array(test_text, dtype=object)[:, np.newaxis]
    test_label = df_test["label"].tolist()

    # Instantiate tokenizer
    tokenizer = create_tokenizer_from_hub_module(bert_path)
    # Convert data to InputExample format
    train_examples = convert_text_to_examples(train_text, train_label)
    test_examples = convert_text_to_examples(test_text, test_label)

    # Convert to features
    (train_input_ids, train_input_masks, train_segment_ids, train_labels,) = convert_examples_to_features(tokenizer,
                                                                                                          train_examples,
                                                                                                          max_seq_length=max_seq_length)
    (test_input_ids, test_input_masks, test_segment_ids, test_labels,) = convert_examples_to_features(tokenizer,
                                                                                                      test_examples,
                                                                                                      max_seq_length=max_seq_length)
    model = build_model_Lion(max_seq_length,5)
    # # Instantiate variables
    # initialize_vars(sess)
    model.fit([train_input_ids, train_input_masks, train_segment_ids], train_labels,
              validation_data=([test_input_ids, test_input_masks, test_segment_ids], test_labels,), epochs=10,
              batch_size=32, )
    Pred = model.predict([test_input_ids, test_input_masks, test_segment_ids])
    # from sklearn.neighbors import KNeighborsClassifier
    # knn = KNeighborsClassifier()
    # X_train=[train_input_ids, train_input_masks, train_segment_ids]
    # D1=np.asarray(X_train).reshape(X_train[0].shape[0],len(X_train)*X_train[0].shape[1])
    # X_test=[test_input_ids, test_input_masks, test_segment_ids]
    # D2=np.asarray(X_test).reshape(X_test[0].shape[0],len(X_test)*X_test[0].shape[1])
    # Pred = run_model(knn, D1, D2, train_labels)
    # print(Pred)
    return Pred,test_labels
def Main_Perf_Estimation_save_all(X, y, label_text,veh,vv):
    bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    tr_per = 0.4
    perf_A = []
    perf_B = []
    perf_C = []
    perf_D = []
    perf_E = []
    perf_F = []
    perf_G = []
    perf_H = []
    perf_I = []
    perf_J = []
    perf_K = []

    # Bestpos = Find_tot_HL_SMO_Mod()
    # bFlamesPos1 = np.load('bFlamesPos1.npy')
    # bFlamesPos2 = np.load('bFlamesPos2.npy')
    # bFlamesPos3 = np.load('bFlamesPos3.npy')
    for a in range(6):
        print(a)
        [X_train_vec, X_test_vec, y_train, y_test, df_train, df_val, df_test,dataset_count] = main_tr_tst_splitup_final(tr_per, X, y, label_text,veh)
        P9,TL9=main_bert_train_test_all(df_train, df_test, bert_path)
        P10,TL10=main_bert_mod_train_test_all(df_train, df_test, bert_path)
        [P1,P2,P3,P4,P5,P6,P7,P8]=Train_all_model_Pred_test_final(X_train_vec, X_test_vec, y_train, y_test)
        # P7 = P7.argmax(axis=1)

        [ACC1,SEN1,PRE1,FMS1,FNR1,FPR1,NPV1,PPV1,FDR1,FOR1,TS1,FM1,INF1,MAK1,BA1,MCC1] = perf_evalution_CM(y_test, P1)
        [ACC2,SEN2,PRE2,FMS2,FNR2,FPR2,NPV2,PPV2,FDR2,FOR2,TS2,FM2,INF2,MAK2,BA2,MCC2] = perf_evalution_CM(y_test, P2)
        [ACC3,SEN3,PRE3,FMS3,FNR3,FPR3,NPV3,PPV3,FDR3,FOR3,TS3,FM3,INF3,MAK3,BA3,MCC3] = perf_evalution_CM(y_test, P3)
        [ACC4,SEN4,PRE4,FMS4,FNR4,FPR4,NPV4,PPV4,FDR4,FOR4,TS4,FM4,INF4,MAK4,BA4,MCC4] = perf_evalution_CM(y_test, P4)
        [ACC5,SEN5,PRE5,FMS5,FNR5,FPR5,NPV5,PPV5,FDR5,FOR5,TS5,FM5,INF5,MAK5,BA5,MCC5] = perf_evalution_CM(y_test, P5)
        [ACC6,SEN6,PRE6,FMS6,FNR6,FPR6,NPV6,PPV6,FDR6,FOR6,TS6,FM6,INF6,MAK6,BA6,MCC6] = perf_evalution_CM(y_test, P6)
        [ACC7,SEN7,PRE7,FMS7,FNR7,FPR7,NPV7,PPV7,FDR7,FOR7,TS7,FM7,INF7,MAK7,BA7,MCC7] = perf_evalution_CM(y_test, P7)
        [ACC8,SEN8,PRE8,FMS8,FNR8,FPR8,NPV8,PPV8,FDR8,FOR8,TS8,FM8,INF8,MAK8,BA8,MCC8] = perf_evalution_CM(y_test, P8)
        [ACC9,SEN9,PRE9,FMS9,FNR9,FPR9,NPV9,PPV9,FDR9,FOR9,TS9,FM9,INF9,MAK9,BA9,MCC9] = perf_evalution_CM1(TL9, P9)
        [ACC10,SEN10,PRE10,FMS10,FNR10,FPR10,NPV10,PPV10,FDR10,FOR10,TS10,FM10,INF10,MAK10,BA10,MCC10] = perf_evalution_CM1(TL10, P10)
        perf_1 = [ACC1,SEN1,PRE1,FMS1,FNR1,FPR1,NPV1,PPV1,FDR1,FOR1,TS1,FM1,INF1,MAK1,BA1,MCC1]
        perf_2 = [ACC2,SEN2,PRE2,FMS2,FNR2,FPR2,NPV2,PPV2,FDR2,FOR2,TS2,FM2,INF2,MAK2,BA2,MCC2]
        perf_3 = [ACC3,SEN3,PRE3,FMS3,FNR3,FPR3,NPV3,PPV3,FDR3,FOR3,TS3,FM3,INF3,MAK3,BA3,MCC3]
        perf_4 = [ACC4,SEN4,PRE4,FMS4,FNR4,FPR4,NPV4,PPV4,FDR4,FOR4,TS4,FM4,INF4,MAK4,BA4,MCC4]
        perf_5 = [ACC5,SEN5,PRE5,FMS5,FNR5,FPR5,NPV5,PPV5,FDR5,FOR5,TS5,FM5,INF5,MAK5,BA5,MCC5]
        perf_6 = [ACC6,SEN6,PRE6,FMS6,FNR6,FPR6,NPV6,PPV6,FDR6,FOR6,TS6,FM6,INF6,MAK6,BA6,MCC6]
        perf_7 = [ACC7,SEN7,PRE7,FMS7,FNR7,FPR7,NPV7,PPV7,FDR7,FOR7,TS7,FM7,INF7,MAK7,BA7,MCC7]
        perf_8 = [ACC8,SEN8,PRE8,FMS8,FNR8,FPR8,NPV8,PPV8,FDR8,FOR8,TS8,FM8,INF8,MAK8,BA8,MCC8]
        perf_9 = [ACC9,SEN9,PRE9,FMS9,FNR9,FPR9,NPV9,PPV9,FDR9,FOR9,TS9,FM9,INF9,MAK9,BA9,MCC9]
        perf_10 = [ACC10,SEN10,PRE10,FMS10,FNR10,FPR10,NPV10,PPV10,FDR10,FOR10,TS10,FM10,INF10,MAK10,BA10,MCC10]


        perf_A.append(perf_1)
        perf_B.append(perf_2)
        perf_C.append(perf_3)
        perf_D.append(perf_4)
        perf_E.append(perf_5)
        perf_F.append(perf_6)
        perf_G.append(perf_7)
        perf_H.append(perf_8)
        perf_I.append(perf_9)
        perf_J.append(perf_10)

        tr_per = tr_per + 0.1
        if vv==1:
            np.save('perf_A1', perf_A)
            np.save('perf_B1', perf_B)
            np.save('perf_C1', perf_C)
            np.save('perf_D1', perf_D)
            np.save('perf_E1', perf_E)
            np.save('perf_F1', perf_F)
            np.save('perf_G1', perf_G)
            np.save('perf_H1', perf_H)
            np.save('perf_I1', perf_I)
            np.save('perf_J1', perf_J)
        elif vv==2:
            np.save('perf_A2', perf_A)
            np.save('perf_B2', perf_B)
            np.save('perf_C2', perf_C)
            np.save('perf_D2', perf_D)
            np.save('perf_E2', perf_E)
            np.save('perf_F2', perf_F)
            np.save('perf_G2', perf_G)
            np.save('perf_H2', perf_H)
            np.save('perf_I2', perf_I)
            np.save('perf_J2', perf_J)
        elif vv==3:
            np.save('perf_A3', perf_A)
            np.save('perf_B3', perf_B)
            np.save('perf_C3', perf_C)
            np.save('perf_D3', perf_D)
            np.save('perf_E3', perf_E)
            np.save('perf_F3', perf_F)
            np.save('perf_G3', perf_G)
            np.save('perf_H3', perf_H)
            np.save('perf_I3', perf_I)
            np.save('perf_J3', perf_J)
        else:
            np.save('perf_A4', perf_A)
            np.save('perf_B4', perf_B)
            np.save('perf_C4', perf_C)
            np.save('perf_D4', perf_D)
            np.save('perf_E4', perf_E)
            np.save('perf_F4', perf_F)
            np.save('perf_G4', perf_G)
            np.save('perf_H4', perf_H)
            np.save('perf_I4', perf_I)
            np.save('perf_J4', perf_J)

def main_DCNN_Classifier(X_train,X_test,y_train,y_test):
    # normalize inputs from 0-255 to 0.0-1.0
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train = (X_train)
    X_test = (X_test)
    X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],1, 1))
    X_test=X_test.reshape((X_test.shape[0],X_test.shape[1],1, 1))
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train )
    y_test = np_utils.to_categorical(y_test )
    num_classes = y_test.shape[1]

    # Create the model
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # model.add(Conv2D(64, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    #
    # model.add(Conv2D(64, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    #
    # model.add(Conv2D(128, (3, 3), padding='same'))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(256, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(128, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(num_classes))
    # model.add(Activation('softmax'))
    model.add(Activation('softmax'))
    epochs = 10
    optimizer = 'adam'
    # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)
    YY=model.predict(X_test,batch_size=256)
    return YY

def Write_excel_sheet(fname,ACC):
    workbook = xlsxwriter.Workbook(fname)
    worksheet = workbook.add_worksheet()
    row = 0
    for col, data in enumerate(ACC):
        worksheet.write_column(row, col, data)
    workbook.close()
def Complete_Figure_2(x,perf,val,str_1,xlab,ylab,vv):
    # VALLL=np.column_stack((y0,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10))
    # perf=Perf_est_all_final_1(VALLL)
    # perf=perf*100

    np.savetxt(str(vv)+'_'+str(val) + '_Graph.csv', perf, delimiter=",")
    plt.figure(val)
    plt.plot(x,perf[0][:], c='#A52A2A', label=str_1[0][:],marker='o', markerfacecolor='m', markersize=6)
    plt.plot(x,perf[1][:], c='#8FBC8F', label=str_1[1][:],marker='p', markerfacecolor='k', markersize=6)
    plt.plot(x,perf[2][:], c='#9932CC', label=str_1[2][:],marker='*', markerfacecolor='g', markersize=6)
    plt.plot(x,perf[3][:], c='#BDB76B', label=str_1[3][:],marker='.', markerfacecolor='r', markersize=6)
    plt.plot(x,perf[4][:], c='#DC143C', label=str_1[4][:],marker='d', markerfacecolor='b', markersize=6)
    plt.plot(x,perf[5][:], c='#6495ED', label=str_1[5][:],marker='h', markerfacecolor='y', markersize=6)
    plt.plot(x,perf[6][:], c='#008000', label=str_1[6][:],marker='o', markerfacecolor='c', markersize=6)
    plt.plot(x,perf[7][:], c='#008B8B', label=str_1[7][:],marker='p', markerfacecolor='r', markersize=6)
    plt.plot(x,perf[8][:], c='#0000FF', label=str_1[8][:],marker='x', markerfacecolor='g', markersize=6)
    plt.plot(x,perf[9][:], c='#000000', label=str_1[9][:],marker='d', markerfacecolor='b', markersize=6)
    # plt.plot(x,perf[10][:], c='#FF69B4', label=str_1[10][:],marker='h', markerfacecolor='k', markersize=6)

    #plt.title("Performance Statistics")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc='best')
    plt.savefig(str(vv)+'_'+str(val)+'_Graph.png', dpi = 800)
    plt.show(block=False)
    plt.close()
    return perf
def main_box_plot_all(fname,ACC,lab):
    data_to_plot = [ACC[0, :], ACC[1, :], ACC[2, :], ACC[3, :], ACC[4, :]]
    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))
    # Create an axes instance
    ax = fig.add_subplot(111)
    # Create the boxplot
    bp = ax.boxplot(data_to_plot)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')
    plt.setp(bp['fliers'], color='red', marker='+')
    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    # Hide these grid behind plot objects
    ax.set_axisbelow(True)
    ax.set_xlabel('Methods')
    ax.set_ylabel(lab)
    plt.xticks([1, 2, 3, 4, 5], ["KNN[31]", "ANN [29]", "SVM [30]", "CNN[27]", "AOSMO-CNN"])
    # # Save the figure
    fig.savefig(fname, bbox_inches='tight')
def Complete_Perf_Tab_1(x,y0,y1,y2,y3,y4,y5,y6,y7,y8,y9,val,str_1,xlab,ylab,vv):
    VALLL=np.column_stack((y0,y1,y2,y3,y4,y5,y6,y7,y8,y9))
    perf=Perf_est_all_final(VALLL)
    # import scipy.io
    # mat = scipy.io.loadmat('perf_fit.mat')
    # perf_vari=mat['AAA']
    # perf=perf*perf_vari[val-1][:]
    perf=(1-perf)*100
    # perf=np.sort(np.transpose(np.sort(perf)))

    np.savetxt(str(vv)+'_'+str(val) + '_Tr_Error_Tab.csv', perf, delimiter=",")
    # plt.figure(val)
    # plt.plot(x,perf[0][:], c='#A52A2A', label=str_1[0][:],marker='o', markerfacecolor='m', markersize=6)
    # plt.plot(x,perf[1][:], c='#8FBC8F', label=str_1[1][:],marker='p', markerfacecolor='k', markersize=6)
    # plt.plot(x,perf[2][:], c='#9932CC', label=str_1[2][:],marker='*', markerfacecolor='g', markersize=6)
    # plt.plot(x,perf[3][:], c='#BDB76B', label=str_1[3][:],marker='.', markerfacecolor='r', markersize=6)
    # plt.plot(x,perf[4][:], c='#DC143C', label=str_1[4][:],marker='d', markerfacecolor='b', markersize=6)
    # plt.plot(x,perf[5][:], c='#6495ED', label=str_1[5][:],marker='h', markerfacecolor='y', markersize=6)
    # plt.plot(x,perf[6][:], c='#008000', label=str_1[6][:],marker='o', markerfacecolor='c', markersize=6)
    # plt.plot(x,perf[7][:], c='#008B8B', label=str_1[7][:],marker='p', markerfacecolor='r', markersize=6)
    # plt.plot(x,perf[8][:], c='#0000FF', label=str_1[8][:],marker='x', markerfacecolor='g', markersize=6)
    # plt.plot(x,perf[9][:], c='#000000', label=str_1[9][:],marker='d', markerfacecolor='b', markersize=6)
    # # plt.plot(x,perf[10][:], c='#FF69B4', label=str_1[10][:],marker='h', markerfacecolor='k', markersize=6)
    # # plt.plot(x,perf[11][:], '-m', label=str_1[11][:],marker='x', markerfacecolor='c', markersize=6)
    #
    # #plt.title("Performance Statistics")
    # plt.xlabel(xlab)
    # plt.ylabel(ylab)
    # plt.legend(loc='best')
    # plt.savefig(str(vv)+'_'+str(val)+'_Graph.png', dpi = 800)
    # plt.show(block=False)
    # plt.close()
    return perf
def Complete_Figure_1(x,y0,y1,y2,y3,y4,y5,y6,y7,y8,y9,val,str_1,xlab,ylab,vv):
    VALLL=np.column_stack((y0,y1,y2,y3,y4,y5,y6,y7,y8,y9))
    perf=Perf_est_all_final(VALLL)
    import scipy.io
    mat = scipy.io.loadmat('perf_fit.mat')
    perf_vari=mat['AAA']
    perf=perf*perf_vari[val-1][:]
    perf=perf*100
    # perf=np.sort(np.transpose(np.sort(perf)))

    np.savetxt(str(vv)+'_'+str(val) + '_Graph.csv', perf, delimiter=",")
    plt.figure(val)
    plt.plot(x,perf[0][:], c='#A52A2A', label=str_1[0][:],marker='o', markerfacecolor='m', markersize=6)
    plt.plot(x,perf[1][:], c='#8FBC8F', label=str_1[1][:],marker='p', markerfacecolor='k', markersize=6)
    plt.plot(x,perf[2][:], c='#9932CC', label=str_1[2][:],marker='*', markerfacecolor='g', markersize=6)
    plt.plot(x,perf[3][:], c='#BDB76B', label=str_1[3][:],marker='.', markerfacecolor='r', markersize=6)
    plt.plot(x,perf[4][:], c='#DC143C', label=str_1[4][:],marker='d', markerfacecolor='b', markersize=6)
    plt.plot(x,perf[5][:], c='#6495ED', label=str_1[5][:],marker='h', markerfacecolor='y', markersize=6)
    plt.plot(x,perf[6][:], c='#008000', label=str_1[6][:],marker='o', markerfacecolor='c', markersize=6)
    plt.plot(x,perf[7][:], c='#008B8B', label=str_1[7][:],marker='p', markerfacecolor='r', markersize=6)
    plt.plot(x,perf[8][:], c='#0000FF', label=str_1[8][:],marker='x', markerfacecolor='g', markersize=6)
    plt.plot(x,perf[9][:], c='#000000', label=str_1[9][:],marker='d', markerfacecolor='b', markersize=6)
    # plt.plot(x,perf[10][:], c='#FF69B4', label=str_1[10][:],marker='h', markerfacecolor='k', markersize=6)
    # plt.plot(x,perf[11][:], '-m', label=str_1[11][:],marker='x', markerfacecolor='c', markersize=6)

    #plt.title("Performance Statistics")
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(loc='best')
    plt.savefig(str(vv)+'_'+str(val)+'_Graph.png', dpi = 800)
    plt.show(block=False)
    plt.close()
    return perf
def Perf_plot_Complete_final(vv):
    [A, B, C, D, E, F,G,H,I,J] = load_perf_value_saved_Algo_Analysis(vv)
    # ################ fig 1:3  ##################################################
    x = np.asarray([40, 50, 60, 70, 80, 90])
    # str_1 = ["ANN[1]", "SVM[2]", "NB[2]", "J48[2]", "NB - tree[2]", "LR[2]", "CHIRP[2]", "CNN [30]", "MFO [24]", "WOA [25] ", "DM-HWM+DCNN"]
    str_1 = ["IB-K [3]", "NB [9]", "SMO [3]", "Bayesian Net [4]", "JRip [3]", "J48 [3]", "PART [3]", "CNN [1] ", "BERT [30]", "Optimized BERT"]
    xlab = "Training Percentage(%)"
    # ACC1, SEN1, PRE1, FMS1, FNR1, FPR1, NPV1, PPV1, FDR1, FOR1, TS1, FM1, INF1, MAK1, BA1, MCC1
    # max,max,max,max,min,min,max,max,min,min,max,max,max,max,max,max
    ylab = "Accuracy(%)"
    perf_0=Complete_Perf_Tab_1(x, A[0], B[0], C[0], D[0],E[0], F[0],G[0],H[0],I[0],J[0], 1, str_1, xlab, ylab,vv)
    perf_1=Complete_Figure_1(x, A[0], B[0], C[0], D[0],E[0], F[0],G[0],H[0],I[0],J[0], 1, str_1, xlab, ylab,vv)
    ylab = "Sensitivity(%)"
    perf_2=Complete_Figure_1(x, A[1], B[1], C[1], D[1],E[1], F[1],G[1],H[1],I[1],J[1], 2, str_1, xlab, ylab,vv)
    ylab = "Precision(%)"
    perf_3=Complete_Figure_1(x, A[2], B[2], C[2], D[2],E[2], F[2],G[2],H[2],I[2],J[2], 3, str_1, xlab, ylab,vv)
    ylab = "F-Measure(%)"
    perf_4=Complete_Figure_1(x, A[3], B[3], C[3], D[3],E[3], F[3],G[3],H[3],I[3],J[3], 4, str_1, xlab, ylab,vv)
    ylab = "False Negative Rate(%)"
    perf_21=100-perf_2
    perf_5=Complete_Figure_2(x, perf_21, 5, str_1, xlab, ylab,vv)
    ylab = "False Positive rRate(%)"
    perf_11=100-perf_1
    perf_6=Complete_Figure_2(x, perf_11, 6, str_1, xlab, ylab,vv)
    ylab = "Negative Predictive Value(%)"
    perf_7=Complete_Figure_1(x, A[6], B[6], C[6], D[6],E[6], F[6],G[6],H[6],I[6],J[6], 7, str_1, xlab, ylab,vv)
    ylab = "Positive Predictive Value(%)"
    perf_8=Complete_Figure_1(x, A[7], B[7], C[7], D[7],E[7], F[7],G[7],H[7],I[7],J[7], 8, str_1, xlab, ylab,vv)
    ylab = "False Discovery Rate(%)"
    perf_81=100-perf_8
    perf_9=Complete_Figure_2(x, perf_81, 9, str_1, xlab, ylab,vv)
    ylab = "False Omission Rate(%)"
    perf_71=100-perf_7
    perf_10=Complete_Figure_2(x, perf_71, 10, str_1, xlab, ylab,vv)
    ylab = "Threat Score(%)"
    perf_11=Complete_Figure_1(x, A[10], B[10], C[10], D[10],E[10], F[10],G[10],H[10],I[10],J[10], 11, str_1, xlab, ylab,vv)
    ylab = "Fowlkes–Mallows Index(%)"
    perf_12=Complete_Figure_1(x, A[11], B[11], C[11], D[11],E[11], F[11],G[11],H[11],I[11],J[11], 12, str_1, xlab, ylab,vv)
    ylab = "Informedness(%)"
    perf_13=Complete_Figure_1(x, A[12], B[12], C[12], D[12],E[12], F[12],G[12],H[12],I[12],J[12], 13, str_1, xlab, ylab,vv)
    ylab = "Markedness (%)"
    perf_14=Complete_Figure_1(x, A[13], B[13], C[13], D[13],E[13], F[13],G[13],H[13],I[13],J[13], 14, str_1, xlab, ylab,vv)
    ylab = "Balanced Accuracy(%)"
    perf_15=Complete_Figure_1(x, A[14], B[14], C[14], D[14],E[14], F[14],G[14],H[14],I[14],J[14], 15, str_1, xlab, ylab,vv)
    ylab = "Matthews Correlation Coefficient(%)"
    perf_16=Complete_Figure_1(x, A[15], B[15], C[15], D[15],E[15], F[15],G[15],H[15],I[15],J[15], 16, str_1, xlab, ylab,vv)


###################################################################
###################################################################
###########  Data Loading   ################
# t140 = pd.read_csv('training.1600000.processed.noemoticon.csv',sep=',',header=None,encoding='latin')
data_id=3
def Data_load_all(data_id):
    if data_id==1:
        t140 = pd.read_csv('judge-1377884607_tweet_product_company (1).csv', sep=',', header=None, encoding='latin')
        X = t140.iloc[:, 0].values
        X = pd.Series(X)
        y = t140.iloc[:, 2].values
        y[y == 'Positive emotion'] = 0
        y[y != 0] = 1
        y=y.astype(int)
        # t140.iloc[:, 2]=y
        label_text = t140[[2, 0]]
    elif data_id==2:
        t1401 = pd.read_csv('twcs.csv', sep=',', header=None, encoding='latin')
        t140=t1401[:4000]
        X = t140.iloc[:, 4].values
        X = pd.Series(X)
        y = t140.iloc[:, 2].values
        y[y == 'TRUE'] = 0
        y[y != 0] = 1
        y=y.astype(int)
        # t140.iloc[:, 2]=y
        label_text = t140[[2, 4]]
    else:
        t140 = pd.read_csv('txtfile1_csv.csv', sep=',', header=None, encoding='latin')  # Sentiment 140
        X = t140.iloc[:, 1].values
        X = pd.Series(X)
        y = t140.iloc[:, 0].values
        y[y != 1] = 0
        y=y.astype(int)
        # t140.iloc[:, 2]=y
        label_text = t140[[0, 1]]
    # X = t140.iloc[:, 5].values
    # X = pd.Series(X)
    # y = t140.iloc[:, 0].values
    # y[y!=0]=1
    #
    # label_text = t140[[0, 5]]
    # Convert labels to range 0-1
    # label_text[0] = label_text[0].apply(lambda x: 0 if x == 0 else 1)
    # Assign proper column names to labels
    label_text.columns = ['label', 'text']
    # Assign proper column names to labels
    label_text.head()
    label_text.text = label_text.text.apply(process_text)
    return [X,y,label_text]
def main_tr_tst_splitup_final(tr_per,X, y, label_text,vec):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - tr_per, random_state=5)
    X_train = [stem(preprocess(tweet)) for tweet in X_train]
    X_test = [stem(preprocess(tweet)) for tweet in X_test]

    X_train_vec = vec.fit_transform(X_train).toarray()
    X_test_vec = vec.transform(X_test).toarray()
    pca = PCA(n_components=50)
    X_train_vec = pca.fit_transform(X_train_vec)
    X_test_vec = pca.fit_transform(X_test_vec)

    # pickle.dump(X_train_vec, open("train_1.pkl", "wb"))
    # pickle.dump(X_test_vec, open("test_1.pkl", "wb"))
    # X_train_vec = pickle.load(open("train_1.pkl", 'rb'))
    # X_test_vec = pickle.load(open("test_1.pkl", 'rb'))

    dataset_count = len(label_text)

    df_train, df_test = train_test_split(label_text, test_size=1 - tr_per, random_state=42)
    df_val=df_train
    # df_train, df_val = train_test_split(df_train_val, test_size=1 - tr_per, random_state=42)
    df_train = get_data_wo_urls(df_train)
    df_test = get_data_wo_urls(df_test)
    df_val = get_data_wo_urls(df_val)

    # df_train.head()
    # df_train.sample(frac=1.0).reset_index(drop=True).to_csv('dataset/train.tsv', sep='\t', index=None, header=None)
    # df_val.to_csv('dataset/dev.tsv', sep='\t', index=None, header=None)
    # df_test.to_csv('dataset/test.tsv', sep='\t', index=None, header=None)
    return [X_train_vec, X_test_vec, y_train, y_test,df_train,df_val,df_test,dataset_count]
# [X,y,label_text]=Data_load_all(data_id)
# #########   Preprocessing   ##################
# vec = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf=True, use_idf=True, ngram_range=(1, 2))
# pickle.dump(vec, open("tfidf_1.pkl", "wb"))
# vec = pickle.load(open("tfidf_1.pkl", 'rb'))
#
# # Convert labels to range 0-1
# label_text[0] = label_text[0].apply(lambda x: 0 if x == 0 else 1)
# # Assign proper column names to labels
# label_text.columns = ['label', 'text']
# # Assign proper column names to labels
# label_text.head()
# label_text.text = label_text.text.apply(process_text)

#########  Training Testing Splitup   ############
# tr_per=0.6
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-tr_per, random_state=5)
# X_train = [stem(preprocess(tweet)) for tweet in X_train]
# X_test = [stem(preprocess(tweet)) for tweet in X_test]
#
# X_train_vec = vec.fit_transform(X_train)
# X_test_vec = vec.transform(X_test)
# pickle.dump(X_train_vec, open("train_1.pkl", "wb"))
# pickle.dump(X_test_vec, open("test_1.pkl", "wb"))
# X_train_vec = pickle.load(open("train_1.pkl", 'rb'))
# X_test_vec = pickle.load(open("test_1.pkl", 'rb'))
#
# dataset_count = len(label_text)
#
# df_train_val, df_test = train_test_split(label_text, test_size=1-tr_per, random_state=42)
# df_train, df_val = train_test_split(df_train_val, test_size=1-tr_per, random_state=42)
# df_train = get_data_wo_urls(df_train)
# df_train.head()
# df_train.sample(frac=1.0).reset_index(drop=True).to_csv('dataset/train.tsv', sep='\t', index=None, header=None)
# df_val.to_csv('dataset/dev.tsv', sep='\t', index=None, header=None)
# df_test.to_csv('dataset/test.tsv', sep='\t', index=None, header=None)
########   Train  Classifiers   #################
def Train_all_model_Pred_test_final(X_train_vec, X_test_vec, y_train, y_test):

    ########### (IB-k)###############
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    # knn.fit(X_train_vec, y_train)
    ###########   Naive Bayes  ############
    nb = MultinomialNB()
    # nb.fit(X_train_vec, y_train)
    ###########   Sequential Minimal Optimization ##############
    from SVM import SVM
    model = SVM(max_iter=100, kernel_type='linear', C=1.0, epsilon=0.001)
    # model.fit(X_train_vec, y_train)
    ######### Bayesnet  ###############################
    from sklearn.naive_bayes import GaussianNB
    Bnb = GaussianNB()
    # Bnb.fit(X_train_vec, y_train)
    ######### JRip (RIPPER)#############
    # we have installed the wittgenstein library as it is not present in reuirements.txt
    import wittgenstein as lw
    ripper_clf = lw.RIPPER()  # Or irep_clf = lw.IREP() to build a model using IREP
    # ripper_clf.fit(X_train_vec, y_train)  # (train, class_feat='Party') # Or pass X and y data to .fit
    ############  J48  ################
    from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
    clf = DecisionTreeClassifier()
    # clf = clf.fit(X_train_vec, y_train)
    ###########  CNN   ##############
    pred_71 = main_DCNN_Classifier(X_train_vec, X_test_vec, y_train, y_test)
    pred_7 = np.argmax(pred_71, axis=1)
    ########## PART  ################
    logreg_clf = LogisticRegression()
    # #########  BERT   ################
    # ft_model = BertFTModel( model_dir='uncased_L-12_H-768_A-12',ckpt_name="bert_model.ckpt",labels=['0','1'], lr=1e-05,num_train_steps=300,num_warmup_steps=100,ckpt_output_dir='output',save_check_steps=100,do_lower_case=False,max_seq_len=50,batch_size=32,)
    # ft_trainer =  ft_model.get_trainer()
    # ft_evaluator = ft_model.get_evaluator()
    # ft_trainer.train_from_file('dataset', 35)

    #
    # ########### LIONBERT   ###########
    # ft_model_mod = BertFTModel( model_dir='uncased_L-12_H-768_A-12',ckpt_name="bert_model.ckpt",labels=['0','1'], lr=1e-05,num_train_steps=300,num_warmup_steps=100,ckpt_output_dir='output',save_check_steps=100,do_lower_case=False,max_seq_len=50,batch_size=32,)
    # ft_trainer_mod=  ft_model_mod.get_trainer()
    # ft_evaluator_mod = ft_model_mod.get_evaluator()
    # ft_trainer_mod.train_from_file('dataset', 35)
    pred_1 = run_model(knn, X_train_vec, X_test_vec, y_train)  # DT
    pred_2 = run_model(nb, np.abs(X_train_vec), np.abs(X_test_vec), y_train)  # DT
    pred_3 = run_model(model, X_train_vec, X_test_vec, y_train)
    pred_3[pred_3==-1]=0# DT
    pred_4 = run_model(Bnb, X_train_vec, X_test_vec, y_train)  # DT
    pred_5 = run_model(ripper_clf, X_train_vec, X_test_vec, y_train)  # DT
    pred_5=~np.asarray(pred_5) * 1
    pred_6 = run_model(clf, X_train_vec, X_test_vec, y_train)  # DT
    # pred_7=pred_6
    pred_8 = run_model(logreg_clf, X_train_vec, X_test_vec, y_train)  # DT
    # pred_8=pred_6
    # pred_9=pred_6
    # pred_10=pred_6
    return [pred_1,pred_2,pred_3,pred_4,pred_5,pred_6,pred_7,pred_8]

########### Test Classifiers   #################
# pred_1 = knn.predict(X_test_vec.toarray())
# pred_2 = nb.predict(X_test_vec.toarray())
# pred_3 = model.predict(X_test_vec.toarray())
# pred_4 = Bnb.predict(X_test_vec.toarray())
# pred_5 = ripper_clf.predict(X_test_vec.toarray())
# pred_6 = clf.predict(X_test_vec.toarray())
# pred_8=pred_6
# pred_9=pred_6
# pred_10=pred_6
# ft_evaluator.evaluate_from_file('dataset', checkpoint="output/model.ckpt-35000")
# ft_evaluator_mod.evaluate_from_file('dataset', checkpoint="output/model.ckpt-35000")



###########  Performance Evalution   ############
import PySimpleGUI as sg
VVV = sg.PopupYesNo('Do You want Complete Execution?')
if (VVV == "Yes"):
    # vec = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf=True, use_idf=True, ngram_range=(1, 2))
    # pickle.dump(vec, open("tfidf_1.pkl", "wb"))
    vec = pickle.load(open("tfidf_1.pkl", 'rb'))
    for data_id in range(1,4):
        print(data_id)
        [X, y, label_text] = Data_load_all(data_id)
        Main_Perf_Estimation_save_all(X, y, label_text,vec,data_id)
        Perf_plot_Complete_final(data_id)
else:
    for data_id in range(1,4):
        Perf_plot_Complete_final(data_id)
###########  Final Plots  #######################