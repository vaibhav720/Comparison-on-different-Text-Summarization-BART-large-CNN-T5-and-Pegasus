
from transformers import BartTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration, T5Tokenizer, PegasusForConditionalGeneration, PegasusTokenizer
from rouge import Rouge

"""# Declaring the Common data which will be used to compare all agents used
###  Data1 is taken from the Entertainment section of the BBC News.
### Data2 is taken from the politics section of the BBC News.
### Data3 is taken from the Business Section of the BBC News.
"""

data1 = """
Gallery unveils interactive tree

A Christmas tree that can receive text messages has been unveiled at London's Tate Britain art gallery.

The spruce has an antenna which can receive Bluetooth texts sent by visitors to the Tate. The messages will be "unwrapped" by sculptor Richard Wentworth, who is responsible for decorating the tree with broken plates and light bulbs. It is the 17th year that the gallery has invited an artist to dress their Christmas tree. Artists who have decorated the Tate tree in previous years include Tracey Emin in 2002.

The plain green Norway spruce is displayed in the gallery's foyer. Its light bulb adornments are dimmed, ordinary domestic ones joined together with string. The plates decorating the branches will be auctioned off for the children's charity ArtWorks. Wentworth worked as an assistant to sculptor Henry Moore in the late 1960s. His reputation as a sculptor grew in the 1980s, while he has been one of the most influential teachers during the last two decades. Wentworth is also known for his photography of mundane, everyday subjects such as a cigarette packet jammed under the wonky leg of a table.
"""

original_summary1 = """
The messages will be "unwrapped" by sculptor Richard Wentworth, who is responsible for decorating the tree with broken plates and light bulbs.A Christmas tree that can receive text messages has been unveiled at London's Tate Britain art gallery.It is the 17th year that the gallery has invited an artist to dress their Christmas tree.The spruce has an antenna which can receive Bluetooth texts sent by visitors to the Tate.His reputation as a sculptor grew in the 1980s, while he has been one of the most influential teachers during the last two decades.
"""
print(len(data1))
print(len(original_summary1))

data2= """
Talks held on Gibraltar's future

Two days of talks on the future of Gibraltar begin at Jack Straw's country residence later on Wednesday.

Officials at the two-day summit at the foreign secretary's official Kent house, Chevening, will plan a new forum on the Rock's future. In October, Mr Straw and his Spanish counterpart Miguel Moratinos agreed to establish a body that would give Gibraltarians a voice in their future. Most Gibraltarians said in a referendum they wanted to remain British.

Gibraltar's Chief Minister Peter Caruana will represent the British citizens living on the Rock, while Britain's Europe Director Dominick Chilcott will represent the UK. Madrid is being represented by Spain's director general for Europe, Jose Maria Pons. The initiative follows Spain's socialist government's decision to put its long-standing sovereignty ambitions on hold. Gibraltarians rejected plans for the Rock's sovereignty to be shared between Britain and Spain in a referendum organised by Gibraltar government.
 """

original_summary2 = """
Gibraltarians rejected plans for the Rock's sovereignty to be shared between Britain and Spain in a referendum organised by Gibraltar government.Most Gibraltarians said in a referendum they wanted to remain British.In October, Mr Straw and his Spanish counterpart Miguel Moratinos agreed to establish a body that would give Gibraltarians a voice in their future.Officials at the two-day summit at the foreign secretary's official Kent house, Chevening, will plan a new forum on the Rock's future.
"""
print(len(data2))
print(len(original_summary2))

data3= """
Japan narrowly escapes recession

Japan's economy teetered on the brink of a technical recession in the three months to September, figures show.

Revised figures indicated growth of just 0.1% - and a similar-sized contraction in the previous quarter. On an annual basis, the data suggests annual growth of just 0.2%, suggesting a much more hesitant recovery than had previously been thought. A common technical definition of a recession is two successive quarters of negative growth.

The government was keen to play down the worrying implications of the data. "I maintain the view that Japan's economy remains in a minor adjustment phase in an upward climb, and we will monitor developments carefully," said economy minister Heizo Takenaka. But in the face of the strengthening yen making exports less competitive and indications of weakening economic conditions ahead, observers were less sanguine. "It's painting a picture of a recovery... much patchier than previously thought," said Paul Sheard, economist at Lehman Brothers in Tokyo. Improvements in the job market apparently have yet to feed through to domestic demand, with private consumption up just 0.2% in the third quarter.


 """

original_summary3 = """
On an annual basis, the data suggests annual growth of just 0.2%, suggesting a much more hesitant recovery than had previously been thought.A common technical definition of a recession is two successive quarters of negative growth.Revised figures indicated growth of just 0.1% - and a similar-sized contraction in the previous quarter.Japan's economy teetered on the brink of a technical recession in the three months to September, figures show.
"""
print(len(data3))
print(len(original_summary3))

class Agent1_BartLargeCNN:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.agent1_model = BartForConditionalGeneration.from_pretrained(model_name)

    def summarize_model(self, data):
        agent1_input_ids = self.tokenizer.encode(data, return_tensors="pt", max_length=1024, truncation=True)
        agent1_summary_ids = self.agent1_model.generate(agent1_input_ids, max_length=570, min_length=30, num_beams=2, early_stopping=True)
        agent1_summarized_text = self.tokenizer.decode(agent1_summary_ids[0], skip_special_tokens=True)
        return agent1_summarized_text

    def calculate_rouge_score(self, predicted, original_summary):
        rouge = Rouge()
        agent1_scores = rouge.get_scores(predicted, original_summary)

        return agent1_scores

agent1_bart = Agent1_BartLargeCNN()

agent1_predicted_summary1 = agent1_bart.summarize_model(data1)


rouge_prediction1 = agent1_bart.calculate_rouge_score(agent1_predicted_summary1, original_summary1)

print("Original Data")
print(data1)
print("\n Original Summary:")
print(original_summary1)
print("\n Agent1 Bart Generated Summary:")
print(agent1_predicted_summary1)
print("\n Rouge 1 score prediction for the Agent1 Facebook Bart-Large-CNN:")
print(rouge_prediction1)

agent1_predicted_summary2 = agent1_bart.summarize_model(data2)

rouge_prediction2 = agent1_bart.calculate_rouge_score(agent1_predicted_summary2, original_summary2)

print("Original Data")
print(data2)
print("\n Original Summary:")
print(original_summary2)
print("\n Agent1 Bart Generated Summary:")
print(agent1_predicted_summary2)
print("\n Rouge 2 score prediction for the Agent1 Facebook Bart-Large-CNN:")
print(rouge_prediction2)

agent1_predicted_summary3 = agent1_bart.summarize_model(data3)

rouge_prediction3 = agent1_bart.calculate_rouge_score(agent1_predicted_summary3, original_summary3)

print("Original Data")
print(data3)
print("\n Original Summary:")
print(original_summary3)
print("\n Agent1 Bart Generated Summary:")
print(agent1_predicted_summary3)
print("\n Rouge 3 score prediction for the Agent1 Facebook Bart-Large-CNN:")
print(rouge_prediction3)

class Agent2_T5:
    def __init__(self, model_name="t5-large"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.agent2_model = T5ForConditionalGeneration.from_pretrained(model_name)

    def summarize_model(self, text):
        max_input_length = 1024
        max_output_length = 500

        chunks = [text[i:i+max_input_length] for i in range(0, len(text), max_input_length)]
        summarized_chunks = []

        for chunk in chunks:
            agent2_input_ids = self.tokenizer.encode(chunk, return_tensors="pt", max_length=max_input_length, truncation=True)
            agent2_summary_ids = self.agent2_model.generate(agent2_input_ids, max_length=max_output_length, min_length=30, do_sample=False)
            summarized_chunk = self.tokenizer.decode(agent2_summary_ids[0], skip_special_tokens=True)
            summarized_chunks.append(summarized_chunk)

        return " ".join(summarized_chunks)

    def evaluate_rouge_score(self, predicted, original_summary):
        rouge = Rouge()
        rouge_scores = rouge.get_scores(predicted, original_summary)
        return rouge_scores

agent2_t5 = Agent2_T5()

agent2_predicted_summary1 = agent2_t5.summarize_model(data1)

rouge_prediction1 = agent2_t5.evaluate_rouge_score(agent2_predicted_summary1, original_summary1)

print("Original Data")
print(data1)
print("\n Original Summary:")
print(original_summary1)
print("\n Agent2 T5 Generated Summary:")
print(agent2_predicted_summary1)
print("\n Rouge 1 score prediction for the Agent2 T5 Model:")
print(rouge_prediction1)

agent2_predicted_summary2 = agent2_t5.summarize_model(data2)

rouge_prediction2 = agent2_t5.evaluate_rouge_score(agent2_predicted_summary2, original_summary2)

print("Original Data")
print(data2)
print("\n Original Summary:")
print(original_summary2)
print("\n Agent2 T5 Generated Summary:")
print(agent2_predicted_summary2)
print("\n Rouge 2 score prediction for the Agent2 T5 Model:")
print(rouge_prediction2)

agent2_predicted_summary3 = agent2_t5.summarize_model(data3)

rouge_prediction3 = agent2_t5.evaluate_rouge_score(agent2_predicted_summary3, original_summary3)

print("Original Data")
print(data3)
print("\n Original Summary:")
print(original_summary3)
print("\n Agent2 T5 Generated Summary:")
print(agent2_predicted_summary3)
print("\n Rouge 3 score prediction for the Agent2 T5 Model:")
print(rouge_prediction3)

class Agent3_Pegasus_large:
    def __init__(self, model_name="google/pegasus-large"):
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)

    def summarize_text(self, text):
        input_ids = self.tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model.generate(input_ids, max_length=150, min_length=30, num_beams=2, early_stopping=True)
        summarized_text = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summarized_text

    def calculate_rouge_score(self, predicted, original_summary):
        rouge = Rouge()
        agent1_scores = rouge.get_scores(predicted, original_summary)
        return agent1_scores

agent3_pegasus_large = Agent3_Pegasus_large()

agent3_predicted_summary1 = agent3_pegasus_large.summarize_text(data1)

rouge_prediction1 = agent3_pegasus_large.calculate_rouge_score(agent3_predicted_summary1, original_summary1)

print("Original Data")
print(data1)
print("\n Original Summary:")
print(original_summary1)
print("\n Agent3 Pegasus Generated Summary:")
print(agent3_predicted_summary1)
print("\n Rouge 1 score prediction for the Agent3 Pegasus:")
print(rouge_prediction1)

agent3_predicted_summary2 = agent3_pegasus_large.summarize_text(data2)

rouge_prediction2 = agent3_pegasus_large.calculate_rouge_score(agent3_predicted_summary2, original_summary2)

print("Original Data")
print(data2)
print("\n Original Summary:")
print(original_summary2)
print("\n Agent3 Pegasus Generated Summary:")
print(agent3_predicted_summary2)
print("\n Rouge 2 score prediction for the Agent3 Pegasus:")
print(rouge_prediction2)

agent3_predicted_summary3 = agent3_pegasus_large.summarize_text(data3)

rouge_prediction3 = agent3_pegasus_large.calculate_rouge_score(agent3_predicted_summary3, original_summary3)

print("Original Data")
print(data3)
print("\n Original Summary:")
print(agent3_predicted_summary3)
print("\n Agent3 Pegasus Generated Summary:")
print(agent1_predicted_summary3)
print("\n Rouge 3 score prediction for the Agent3 Pegasus:")
print(rouge_prediction3)

"""## References
### 1) https://huggingface.co/facebook/bart-large-cnn
### 2) https://huggingface.co/google-t5/t5-large
### 3) https://huggingface.co/google/pegasus-large
### 4) https://www.kaggle.com/datasets/pariza/bbc-news-summary
"""