#just making an AI chatbot who will chat with people in instagram
#and learn from them how to talk by collecting data from them and storing it as a json file
#as of now I am making him to collect data after that I will train him manually
#i guess using the bot on discord will be better as of now
#first test succcesful
#rest of the code will be written afterwards
#gotta stem the words too
#Upgraded version of ada is in google colab as chatbot_experiment
#import torch
#import torch.nn as nn
#from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords
#import numpy as np
import random
#import json
import discord
import wikipedia

import openai

openai.api_key = '<api_key>'


chat_data  = {"userdata":[],"ada_data":[]}

class My_client(discord.Client):
    async def on_ready(self):
        print("logged on as ",self.user)

    async def on_message(self,message):
        if message.author!= self.user:
            
            d = message.content
            d = d.lower()
            if 'ada' in d:
                D = d.replace('ada','')
                resp = response = openai.Completion.create(
                                      engine="text-davinci-002",
                                      prompt=D,
                                      temperature=0.3,
                                      top_p=1.0,
                                      max_tokens=150,
                                      frequency_penalty=0.5,
                                      presence_penalty=0.0,
                                      stop=["You:"]
)
                response = resp['choices'][0]['text']
                chat_data['userdata'].append(d)
                chat_data['ada_data'].append(response) 

                print(f'{message.author}:',d)
                print('Ada:',response)
                
                                    
                await message.channel.send(f"{message.author.mention} {response}")

#    async def send_user_data(self,question,message):

#       if user_data['userdata'][0]['patterns']:

 #           await message.channel.send(f"{message.author.mention} {question}")


            






while True:
    client = My_client()
    client.run('ODQ2MzgyMTM4MzA3NjQxMzY0.YKusyg._0G0v9fVBqluUBIdiJwn5-s92yc')
    j = json.dumps(chat_data)

#    with open('chat_data.txt','wb') as f:
#        f.write(j)
#        f.close()
        




   
'''
while True:
    

    d = str(input(">>>>"))
    if d == 'exit':
        break
    D = d
    
    d = analyze(d,all_words)

    d = torch.tensor(d,dtype = torch.float32)

    out = model(d)
            #Old method
            #val = round(model(d).item())

    _,val = torch.max(out,dim = 1)
      
        
    for intent in intents['intents']:
        if tags[val.item()] == intent['tag']:
            print("AI:",random.choice(intent['response']))


    #algorithm for data learning

    #I have to store user questions and responses

 #   if check_pattern(D,intents) == None:
  #      user_data['userdata'][0]['patterns'].append(str(D))
   # print(check_pattern(D,intents))
  #  print(user_data['userdata'][0]['patterns'])


n1 = int(input('>>>'))
if n1 == 1:
    torch.save(model,'Ada_retrained.pth')


j = json.dumps(user_data)

with open('userdata1.json','w') as f:
    f.write(j)
    f.close()



#users.append('met users')

#user = random.choice(users)

#question = random.choice(user_data['userdata'][0]['patterns'])

#ask question to get responses
#d = read response
#userdata['userdata'][0]['responses'].append(d)

#if check_response(D,intents) == False:
#   user_data['userdata'][0]['responses'].append(D)

    


#now have to split data into questions and responses
#have to randomly sample questions from user data to get responses for the data


    
#'''       


            


                    
