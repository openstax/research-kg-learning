import unittest
import spacy
from data_processing_utils import get_closest_match, tag_terms, insert_relation_tags, add_relation
#insert_relation_tags, add_relation, read_spacy_docs


class TestDataProcessingUtils(unittest.TestCase):
    
    def test_get_closet_matches(self):
        
        indices1 = [(12, 15)]
        indices2 = [(18, 20)]
        
        ix = get_closest_match(indices1, indices2)
        self.assertEqual(ix, (indices1[0], indices2[0]))
        
        indices1 = [(12, 15), (5, 6)]
        indices2 = [(8, 10)]
        
        ix = get_closest_match(indices1, indices2)
        self.assertEqual(ix, (indices1[1], indices2[0]))

    def test_tag_terms(self):

        terms = ['cell', 'cell wall', 'biologist'] 
        text = 'A biologist will tell you that a cell contains a cell wall.'

        found_terms = {"biologist": {"text": ["biologist"], "tag": ["NN"], "indices": [(1, 2)]},
                       "cell": {"text": ["cell"], "tag": ["NN"], "indices": [(7, 8)]},
                       "cell wall": {"text": ["cell wall"], "tag": ["NN NN"], "indices": [(10, 12)]}}
        tokenized_text = ["A", "biologist", "will", "tell", "you", "that", "a", "cell", "contains",
                          "a", "cell", "wall", "."]
        bioes_tags = ["O", "S", "O", "O", "O", "O", "O", "S", "O", "O", "B", "E", "O"]
        solution = (tokenized_text, bioes_tags, found_terms)
        output = tag_terms(text, terms)

        self.assertEqual(output, solution)
        
        terms = ['He', 'helium', 'gas'] 
        text = 'He talks about helium (He), which is a gas or plural gases.'

        found_terms = {"he": {"text": ["He"], "tag": ["PRP"], "indices": [(5, 6)]},
                       "helium": {"text": ["helium"], "tag": ["NN"], "indices": [(3, 4)]},
                       "gas": {"text": ["gas", "gases"], "tag": ["NN", "NNS"], 
                               "indices": [(11, 12), (14, 15)]}}
        tokenized_text = ["He", "talks", "about", "helium", "(", "He", ")", ",", "which", "is",
                          "a", "gas", "or", "plural", "gases", "."]
        bioes_tags = ["O", "O", "O", "S", "O", "S", "O", "O", "O", "O", "O", "S", "O", "O", "S", "O"]
        solution = (tokenized_text, bioes_tags, found_terms)
        output = tag_terms(text, terms)

        self.assertEqual(output, solution)
        
        
    def test_insert_relation_tags(self):
        tokenized_text = ["A", "biologist", "will", "tell", "you", "that", "a", "cell", "contains",
                          "a", "cell", "wall", "."]
        indices = ((1, 2), (10, 12))
        solution = ["A", "<e1>", "biologist", "</e1>", "will", "tell", "you", "that", "a", "cell", 
                    "contains", "a", "<e2>", "cell", "wall", "</e2>", "."]
        output = insert_relation_tags(tokenized_text, indices)
        self.assertEqual(output, solution)
    
    def test_add_relation(self):
        
        # inputs
        relations_db = {
            "has-part": {
                "cell -> cell wall": {
                    "sentences": [],
                    "e1_representations": ["cell"],
                    "e2_representations": ["cell wall"],
                }
            },
            "no-relation": {}
        }
        tokenized_text = ["A", "biologist", "will", "tell", "you", "that", "a", "cell", "contains",
                          "a", "cell", "wall", "."]
        found_terms = {"biologist": {"text": ["biologist"], "tag": ["NN"], "indices": [(1, 2)]},
                       "cell": {"text": ["cell"], "tag": ["NN"], "indices": [(7, 8)]},
                       "cell wall": {"text": ["cell wall"], "tag": ["NN NN"], "indices": [(10, 12)]}}
        
        # solution
        haspart_sentence = ["A", "biologist", "will", "tell", "you", "that", "a", "<e1>", "cell", 
                            "</e1>", "contains", "a", "<e2>", "cell", "wall", "</e2>", "."]
        nr_sent = ["A", "<e1>", "biologist", "</e1>", "will", "tell", "you", "that", 
                    "a", "<e2>", "cell", "</e2>", "contains", "a", "cell", "wall", "."]
        relations_db_solution = {
            "has-part": {
                "cell -> cell wall": {
                    "sentences": [haspart_sentence],
                    "e1_representations": ["cell"],
                    "e2_representations": ["cell wall"],
                }
            },
            "no-relation": {
                "biologist -> cell": {
                    "sentences": [nr_sent],
                    "e1_representations": ["biologist"],
                    "e2_representations": ["cell"]
                }
            }
        }
        
        output = add_relation(("biologist", "cell"), found_terms, tokenized_text, 
                              relations_db)
        self.assertEqual(output["no-relation"], relations_db_solution["no-relation"])
  
if __name__ == "__main__":
    unittest.main()