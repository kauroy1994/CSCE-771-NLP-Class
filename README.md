# CV Analysis and Keyword Extraction Techniques

This notebook demonstrates a comprehensive analysis of extracting keywords from curriculum vitae (CV) texts using several machine learning and symbolic techniques. The key focus is on comparing various methods for identifying the most informative keywords and improving the overall quality of CVs.

## Features

1. **BERT Token Classification**: 
   - Fine-tunes the `bert-base-uncased` model from Huggingface's Transformers library to classify tokens from CV texts and identify key components. 
   - The model is trained across multiple epochs, with loss values tracked to monitor performance.

2. **ConceptNet Knowledge Graph Integration**:
   - Uses the ConceptNet API to extract relevant keywords based on the semantic relationships of words in the CV text. 
   - Ranks keywords based on their connection scores (i.e., the number of links to other concepts in ConceptNet).

3. **TF-IDF Based Keyword Extraction**:
   - Uses Term Frequency-Inverse Document Frequency (TF-IDF) to extract keywords based on their importance within the CV text relative to a larger document corpus. 
   - This method emphasizes terms that are frequent in a specific document but rare in the broader corpus.

4. **RAKE (Rapid Automatic Keyword Extraction)**:
   - Implements RAKE for unsupervised keyword extraction. This technique uses word co-occurrence patterns and word frequency to determine keyword relevance.
   - Suitable for fast keyword extraction without relying on pre-trained models or external APIs.

5. **Gensim-Based Keyword Extraction**:
   - Utilizes Gensim’s keyword extraction method, which leverages topic modeling techniques to extract the most relevant terms in the text.
   - Integrates well with large-scale textual corpora and unsupervised learning scenarios.

6. **Comparison of Techniques**:
   - The notebook provides a comparative analysis of these techniques, focusing on their effectiveness in extracting informative keywords. 
   - Various metrics, such as precision, recall, and overall keyword relevance, are computed to assess the performance of each method.

## Key Components

1. **Training and Fine-Tuning BERT**: 
   - The notebook trains the BERT model on CV data to classify and extract key entities.
   - Loss metrics are reported after each epoch for fine-tuning.

2. **Symbolic Knowledge Graph Integration with ConceptNet**:
   - Keywords are extracted by querying ConceptNet for their relevance and semantic connections to other important concepts.

3. **Statistical and Unsupervised Approaches**:
   - The TF-IDF, RAKE, and Gensim methods are applied to the same CV data to identify keywords. 
   - Results from these statistical and unsupervised approaches are compared to the BERT and ConceptNet-based methods.

4. **Comparative Metrics**:
   - The notebook calculates precision and recall for the extracted keywords based on ground-truth labels. This enables a detailed comparison of the accuracy and relevance of the different extraction methods.

## Dependencies

- `transformers`: For working with pre-trained BERT models.
- `torch`: For model training and evaluation.
- `pandas`: For data manipulation and result visualization.
- `requests`: For accessing ConceptNet API.
- `nltk`: For Natural Language Processing tasks, particularly with RAKE.
- `scikit-learn`: For computing TF-IDF scores.
- `gensim`: For topic modeling and keyword extraction.
  
To install the required dependencies, run the following:
```bash
pip install transformers torch pandas requests nltk scikit-learn gensim
```

## How to Use

1. **Fine-tune the BERT Model**:
   - The notebook fine-tunes BERT on the CV dataset. You can modify the number of epochs, learning rate, and other hyperparameters.

2. **Run Keyword Extraction Techniques**:
   - Each keyword extraction method (BERT, ConceptNet, TF-IDF, RAKE, Gensim) can be run sequentially on the provided CV text.
   - The extracted keywords for each method are compared at the end, using various evaluation metrics.

3. **Compare Results**:
   - After running the extraction methods, compare the results across techniques.
   - Evaluate precision, recall, and other metrics to determine the most effective keyword extraction technique for your data.

## Example Output

1. **Top 3 Keywords Based on ConceptNet**:
   - Displays the top 3 most relevant keywords based on the connection scores from ConceptNet.

2. **Predicted Keywords from BERT**:
   - Provides a list of keywords identified by the fine-tuned BERT model from the test CV text.

3. **Comparison of Techniques**:
   - Shows side-by-side comparisons of the keywords extracted from different methods (BERT, ConceptNet, TF-IDF, RAKE, Gensim).
   - Precision and recall values for each method are displayed, allowing users to identify which approach works best for their CVs.

## Future Directions

- **Extend the dataset**: Apply the techniques to a broader set of CVs to analyze scalability and generalizability.
- **Experiment with additional models**: Explore other pre-trained language models such as GPT or T5 for keyword extraction.
- **Enhance ConceptNet integration**: Incorporate more advanced symbolic knowledge graphs to improve the quality of extracted keywords.

## Author

This notebook is part of the curriculum for **CSCE 771 - Architectures for Natural Language Processing**. The focus is on exploring a variety of keyword extraction techniques and comparing their performance.
