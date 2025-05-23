from rag_tools import  PDFAssertionValidator
import json
# Initialize system with PDF directory
validator = PDFAssertionValidator("../resources/")

# Validate an assertion
result = validator.validate_assertion(
    "The stratum corneum of the esophageal epithelium is characterized by corneocytes embedded in a "
    "lipid-rich extracellular matrix, providing mechanical reinforcement and maintaining essential barrier "
    "functions of the esophageal lining.")

print(json.dumps(result, indent=4))
# Output structure
# {
#     "assertion": "Neural networks require large amounts of training data",
#     "validation": "The statement is generally accurate...",
#     "confidence": 0.8,
#     "sources": [
#         {
#             "source": "ml_textbook.pdf",
#             "page": 142,
#             "text": "Deep learning models typically require datasets..."
#         },
#         {
#             "source": "ai_survey.pdf",
#             "page": 23,
#             "text": "Training data requirements remain a key challenge..."
#         }
#     ]
# }