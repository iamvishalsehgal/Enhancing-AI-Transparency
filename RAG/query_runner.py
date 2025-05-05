from KG_RAG import rag_agent

# Test basic graph queries
print(rag_agent.run("What models are trained on ImageNet-1k?"))
print(rag_agent.run("What are the limitations of FalconSafesFWImageDetection?"))

# Test hybrid reasoning
print(rag_agent.run("Compare the architectures of ResNet50 and MobileNetV3"))
print(rag_agent.run("What datasets are used for training BERT?"))