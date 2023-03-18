from flask import Flask, request, jsonify

from random import randint
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model (weights)
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()

app = Flask(__name__)


def answerGuesser(text):
 
#   text = '[CLS] I want to [MASK] the car because it is cheap . [SEP]'
  print("##################################3")
  print(text)
  tokenized_text = tokenizer.tokenize(text)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  masked_index = tokenized_text.index('[MASK]') 

  # Create the segments tensors.
  segments_ids = [0] * len(tokenized_text)

  # Convert inputs to PyTorch tensors
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segments_ids])

  

  # Predict all tokens
  with torch.no_grad():
      predictions = model(tokens_tensor, segments_tensors)

  predicted_index = torch.argmax(predictions[0, masked_index]).item()
  # print(predicted_index)
  predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
  
  indxs = torch.topk(predictions[0, masked_index], 5).indices
  ansList = []
  for x in indxs:
    predicted_token = tokenizer.convert_ids_to_tokens([x.item()])
    ansList.append(predicted_token[0])
  return ansList

@app.route('/data',methods = ['POST'])
def hello():
    print("getting started")
    # print(request.json)
    data = request.json
    # print(data["data"])
    if not data["data"]: return {"error":"File Empty"}
    file = open("data.txt","w")
    file.write(data["data"])
    return "File Recieved"

@app.route('/gen/<numberofq>/', methods=['GET'])
def generateQ(numberofq):

    filer = open("data.txt")
    para = filer.read()
    if not para: return {"error":"Data file empty, Please send the data from which you want the data to be generated"}
    sentences = para.split(".")
    print(sentences)
    # exit()

    sentenceList = []
    WordMaskList = []
    print("*****************\n Generating {} questions \n*****************".format(numberofq))
    # for ss in sentences:
    for p in range(int(numberofq)):
        ss = sentences[randint(0,len(sentences)-1)]
        ss_list = ss.split(" ")

        rand = randint(0,len(ss_list)-1)
        word_mask = ss_list[rand]
        WordMaskList.append(word_mask)
        idx = ss_list.index(word_mask) 
        newSentence = "[CLS] "

        for id,x in enumerate(ss_list):
            if id == idx:
                newSentence+="[MASK] "
            else:
                newSentence+=x
                newSentence+=" "
        newSentence+="[SEP]"

        sentenceList.append(newSentence)

    QuestionsList = []
    dataFrame = pd.DataFrame(columns=["Question","a","b","c","d"])
    for x,word_mask in zip(sentenceList,WordMaskList): 
        ans = answerGuesser(x)
        if word_mask in ans:
            ans.remove(word_mask)
        question = x.replace("[MASK]","____________").replace("[CLS]","").replace("[SEP]","")
        df = {"Question":question, "a":word_mask,"b":ans[0],"c":ans[1],"d":ans[2]}
        dataFrame = dataFrame.append(df,ignore_index=True)
    
    output = dataFrame.to_json(orient='records')[1:-1].replace('},{', '} {')
    print(output)

    return str(output)