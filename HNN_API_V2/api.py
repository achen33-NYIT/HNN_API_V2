import time
from HopfieldNetwork import Text_Processor 
from flask import Flask, request

default_words = ['BEANS', 'APPLE', 'VALID', 'STACK']
txt_processor = Text_Processor(default_words)


app = Flask(__name__)

@app.route('/time')
def get_current_time():
    return {"time": 22}


@app.route('/set-words', methods=['PUT'])
def set_words_list():
    pass
    
@app.route('/process-word', methods=["POST", "GET"])
def process_word():
    # args = request.args
    # word = args['word']
    # global txt_processor
    data = request.json
    # print(data)
    
    # words_list = ['BEANS', 'APPLE', 'VALID', 'STACK']
    word_pair = txt_processor.process_word(data)
    input_word = word_pair[0]
    output_word = word_pair[1]
    

    NEURAL_NETWORK = [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1]

    API_DATA = {"input": input_word,
            "output": output_word,
            "balls": []}
    
    BALLS_ARR = API_DATA["balls"]

    id = 0
    for num in range(1):
       BALL_DICT = {}
       BALLS_ARR.append(BALL_DICT)
       BALL_DICT["id"] = id
       BALL_DICT["values"] = []
       id += 1
       for NEURON in NEURAL_NETWORK:
               BALL_DICT["values"].append({"val": str(NEURON)})
#    print(len(BALL_DICT["values"]))
               
    # print(API_DATA["balls"])
    return API_DATA
    