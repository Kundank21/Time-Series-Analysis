from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WikipediaLoader

class KnowledgeGraphExtractor:
    def __init__(self, model_name="Babelscape/rebel-large", text_splitter=None):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.text_splitter = text_splitter or RecursiveCharacterTextSplitter(chunk_size=512, length_function=len, is_separator_regex=False)

    def extract_triplets(self, text):
        triplets = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and relation != '' and object_ != '':
            triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
        return triplets

    def from_text_to_kb(self, text, verbose=False):
        class KB:
            def __init__(self):
                self.relations = []

            def are_relations_equal(self, r1, r2):
                return all(r1[attr] == r2[attr] for attr in ["head", "type", "tail"])

            def exists_relation(self, r1):
                return any(self.are_relations_equal(r1, r2) for r2 in self.relations)

            def add_relation(self, r):
                if not self.exists_relation(r):
                    self.relations.append(r)

            def print(self):
                print("Relations:")
                for r in self.relations:
                    print(f"  {r}")

        kb = KB()

        # Tokenize text
        model_inputs = self.tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')

        if verbose:
            print(f"Num tokens: {len(model_inputs['input_ids'][0])}")

        # Generate
        gen_kwargs = {
            "max_length": 216,
            "length_penalty": 0,
            "num_beams": 3,
            "num_return_sequences": 3
        }

        generated_tokens = self.model.generate(
            **model_inputs,
            **gen_kwargs,
        )

        decoded_preds = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

        # Create KB
        for sentence_pred in decoded_preds:
            relations = self.extract_triplets(sentence_pred)
            for r in relations:
                kb.add_relation(r)

        return kb

    def insert_into_neo4j(self, raw_documents, graph, verbose=False):
        for doc in raw_documents:
            kb = self.from_text_to_kb(doc.page_content, verbose=verbose)

            for relation in kb.relations:
                head = relation['head']
                relationship = relation['type']
                tail = relation['tail']

                cypher = (
                    f"MERGE (h:`{head}`)"
                    + f" MERGE (t:`{tail}`)"
                    + f" MERGE (h)-[:`{relationship}`]->(t)"
                )
                print(cypher)
                graph.query(cypher)

        graph.refresh_schema()
