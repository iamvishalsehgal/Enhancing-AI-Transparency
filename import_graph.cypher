// AUTO-GENERATED CYPHER IMPORT
CREATE (mcro_mobilenetv3small100lambin1k:Model {id: "mcro:mobilenetv3small100lambin1k"});
CREATE (mcro_mobilenetv3small100lambin1k_ModelDetail:ModelDetailSection {id: "mcro:mobilenetv3small100lambin1k-ModelDetail"});
CREATE (mcro_mobilenetv3small100lambin1k_Citation:CitationInformationSection {id: "mcro:mobilenetv3small100lambin1k-Citation", hasTextValue: "@inproceedingshoward2019searching,
  title=Searching for mobilenetv3,
  author=Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and others,
  booktitle=Proceedings of the IEEE/CVF international conference on computer vision,
  pages=1314--1324,
  year=2019"});
CREATE (mcro_mobilenetv3small100lambin1k_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:mobilenetv3small100lambin1k-ModelArchitecture", hasTextValue: "Image classification / feature backbone"});
CREATE (mcro_mobilenetv3small100lambin1k_Dataset:DatasetInformationSection {id: "mcro:mobilenetv3small100lambin1k-Dataset", hasTextValue: "ImageNet-1k"});
CREATE (mcro_mobilenetv3small100lambin1k_UseCase:UseCaseInformationSection {id: "mcro:mobilenetv3small100lambin1k-UseCase", hasTextValue: "Image Classification"});
CREATE (mcro_mobilenetv3small100lambin1k_ModelParameter:ModelParameterSection {id: "mcro:mobilenetv3small100lambin1k-ModelParameter"});
CREATE (mcro_allMiniLML6v2:Model {id: "mcro:allMiniLML6v2"});
CREATE (mcro_allMiniLML6v2_UseCaseInformationSection:UseCaseInformationSection {id: "mcro:allMiniLML6v2-UseCaseInformationSection", hasTextValue: "Our model is intended to be used as a sentence and short paragraph encoder. Given an input text, it outputs a vector which captures 
the semantic information. The sentence vector may be used for information retrieval, clustering or sentence similarity tasks.

By default, input text longer than 256 word pieces is truncated."});
CREATE (mcro_allMiniLML6v2_TrainingDataInformationSection:TrainingDataInformationSection {id: "mcro:allMiniLML6v2-TrainingDataInformationSection", hasTextValue: "We use the concatenation from multiple datasets to fine-tune our model. The total number of sentence pairs is above 1 billion sentences.
We sampled each dataset given a weighted probability which configuration is detailed in the `data_config.json` file."});
CREATE (mcro_allMiniLML6v2_ModelArchitectureInformationSection:ModelArchitectureInformationSection {id: "mcro:allMiniLML6v2-ModelArchitectureInformationSection", hasTextValue: "We used the pretrained [`nreimers/MiniLM-L6-H384-uncased`](https://huggingface.co/nreimers/MiniLM-L6-H384-uncased) model and fine-tuned in on a 
1B sentence pairs dataset. We use a contrastive learning objective: given a sentence from the pair, the model should predict which out of a set of randomly sampled other sentences, was actually paired with it in our dataset."});
CREATE (mcro_Falconsainsfwimagedetection:Model {id: "mcro:Falconsainsfwimagedetection"});
CREATE (mcro_Falconsainsfwimagedetection_ModelDetail:ModelDetailSection {id: "mcro:Falconsainsfwimagedetection-ModelDetail", hasTextValue: "google/vit-base-patch16-224-in21k"});
CREATE (mcro_Falconsainsfwimagedetection_UseCase:UseCaseInformationSection {id: "mcro:Falconsainsfwimagedetection-UseCase", hasTextValue: "NSFW Image Classification"});
CREATE (mcro_Falconsainsfwimagedetection_TrainingData:TrainingDataInformationSection {id: "mcro:Falconsainsfwimagedetection-TrainingData", hasTextValue: "80,000 images"});
CREATE (mcro_Falconsainsfwimagedetection_Reference:ReferenceInformationSection {id: "mcro:Falconsainsfwimagedetection-Reference", hasTextValue: "ImageNet-21k Dataset"});
CREATE (mcro_Falconsainsfwimagedetection_License:LicenseInformationSection {id: "mcro:Falconsainsfwimagedetection-License"});
CREATE (mcro_Falconsainsfwimagedetection_Architecture:ModelArchitectureInformationSection {id: "mcro:Falconsainsfwimagedetection-Architecture", hasTextValue: "transformer encoder"});
CREATE (mcro_Falconsainsfwimagedetection_Limitation:Entity {id: "mcro:Falconsainsfwimagedetection-Limitation"});
CREATE (mcro_dima806fairfaceageimagedetection:Model {id: "mcro:dima806fairfaceageimagedetection"});
CREATE (mcro_dima806fairfaceageimagedetection_Performance:PerformanceMetricInformationSection {id: "mcro:dima806fairfaceageimagedetection-Performance", hasTextValue: "59% accuracy"});
CREATE (mcro_bertbasemodeluncased:Model {id: "mcro:bertbasemodeluncased"});
CREATE (mcro_bertbasemodeluncased_ModelDetail:ModelDetailSection {id: "mcro:bertbasemodeluncased-ModelDetail"});
CREATE (mcro_bertbasemodeluncased_Citation:CitationInformationSection {id: "mcro:bertbasemodeluncased-Citation"});
CREATE (mcro_bertbasemodeluncased_Architecture:ModelArchitectureInformationSection {id: "mcro:bertbasemodeluncased-Architecture"});
CREATE (mcro_bertbasemodeluncased_IntendedUseCase:UseCaseInformationSection {id: "mcro:bertbasemodeluncased-IntendedUseCase"});
CREATE (mcro_bertbasemodeluncased_Consideration:ConsiderationInformationSection {id: "mcro:bertbasemodeluncased-Consideration"});
CREATE (mcro_bertbasemodeluncased_Limitation:LimitationInformationSection {id: "mcro:bertbasemodeluncased-Limitation"});
CREATE (mcro_bertbasemodeluncased_TrainingData:TrainingDataInformationSection {id: "mcro:bertbasemodeluncased-TrainingData"});
CREATE (mcro_clip:Model {id: "mcro:clip"});
CREATE (mcro_clip_ModelDetail:ModelDetailSection {id: "mcro:clip-ModelDetail"});
CREATE (mcro_clip_Citation:CitationInformationSection {id: "mcro:clip-Citation", hasTextValue: "- [Blog Post](https://openai.com/blog/clip/)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)"});
CREATE (mcro_clip_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:clip-ModelArchitecture", hasTextValue: "The base model uses a ViT-L/14 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder. These encoders are trained to maximize the similarity of (image, text) pairs via a contrastive loss.

The original implementation had two variants: one using a ResNet image encoder and the other using a Vision Transformer. This repository has the variant with the Vision Transformer."});
CREATE (mcro_clip_UseCase:UseCaseInformationSection {id: "mcro:clip-UseCase"});
CREATE (mcro_clip_PrimaryIntendedUseCase:PrimaryIntendedUseCaseInformationSection {id: "mcro:clip-PrimaryIntendedUseCase", hasTextValue: "The model is intended as a research output for research communities. We hope that this model will enable researchers to better understand and explore zero-shot, arbitrary image classification. We also hope it can be used for interdisciplinary studies of the potential impact of such models - the CLIP paper includes a discussion of potential downstream impacts to provide an example for this sort of analysis."});
CREATE (mcro_clip_OutOfScopeUseCase:OutOfScopeUseCaseSectionInformation {id: "mcro:clip-OutOfScopeUseCase", hasTextValue: "**Any** deployed use case of the model - whether commercial or not - is currently out of scope. Non-deployed use cases such as image search in a constrained environment, are also not recommended unless there is thorough in-domain testing of the model with a specific, fixed class taxonomy. This is because our safety assessment demonstrated a high need for task specific testing especially given the variability of CLIP’s performance with different class taxonomies. This makes untested and unconstrained deployment of the model in any use case currently potentially harmful. 

Certain use cases which would fall under the domain of surveillance and facial recognition are always out-of-scope regardless of performance of the model. This is because the use of artificial intelligence for tasks such as these can be premature currently given the lack of testing norms and checks to ensure its fair use.

Since the model has not been purposefully trained in or evaluated on any languages other than English, its use should be limited to English language use cases."});
CREATE (mcro_clip_Dataset:DatasetInformationSection {id: "mcro:clip-Dataset"});
CREATE (mcro_clip_TrainingData:TrainingDataInformationSection {id: "mcro:clip-TrainingData", hasTextValue: "The model was trained on publicly available image-caption data. This was done through a combination of crawling a handful of websites and using commonly-used pre-existing image datasets such as [YFCC100M](http://projects.dfki.uni-kl.de/yfcc100m/). A large portion of the data comes from our crawling of the internet. This means that the data is more representative of people and societies most connected to the internet which tend to skew towards more developed nations, and younger, male users."});
CREATE (mcro_clip_QuantativeAnalysis:QuantativeAnalysisSection {id: "mcro:clip-QuantativeAnalysis"});
CREATE (mcro_clip_Performance:PerformanceMetricInformationSection {id: "mcro:clip-Performance", hasTextValue: "We have evaluated the performance of CLIP on a wide range of benchmarks across a variety of computer vision datasets such as OCR to texture recognition to fine-grained classification. The paper describes model performance on the following datasets:

- Food101
- CIFAR10   
- CIFAR100   
- Birdsnap
- SUN397
- Stanford Cars
- FGVC Aircraft
- VOC2007
- DTD
- Oxford-IIIT Pet dataset
- Caltech101
- Flowers102
- MNIST   
- SVHN 
- IIIT5K   
- Hateful Memes   
- SST-2
- UCF101
- Kinetics700
- Country211
- CLEVR Counting
- KITTI Distance
- STL-10
- RareAct
- Flickr30
- MSCOCO
- ImageNet
- ImageNet-A
- ImageNet-R
- ImageNet Sketch
- ObjectNet (ImageNet Overlap)
- Youtube-BB
- ImageNet-Vid"});
CREATE (mcro_clip_Consideration:ConsiderationInformationSection {id: "mcro:clip-Consideration"});
CREATE (mcro_clip_Limitation:LimitationInformationSection {id: "mcro:clip-Limitation", hasTextValue: "CLIP and our analysis of it have a number of limitations. CLIP currently struggles with respect to certain tasks such as fine grained classification and counting objects. CLIP also poses issues with regards to fairness and bias which we discuss in the paper and briefly in the next section. Additionally, our approach to testing CLIP also has an important limitation- in many cases we have used linear probes to evaluate the performance of CLIP and there is evidence suggesting that linear probes can underestimate model performance."});
CREATE (mcro_clip_BiasAndFairness:EthicalConsiderationSection {id: "mcro:clip-BiasAndFairness", hasTextValue: "We find that the performance of CLIP - and the specific biases it exhibits - can depend significantly on class design and the choices one makes for categories to include and exclude. We tested the risk of certain kinds of denigration with CLIP by classifying images of people from [Fairface](https://arxiv.org/abs/1908.04913) into crime-related and non-human animal categories. We found significant disparities with respect to race and gender. Additionally, we found that these disparities could shift based on how the classes were constructed. (Details captured in the Broader Impacts Section in the paper).

We also tested the performance of CLIP on gender, race and age classification using the Fairface dataset (We default to using race categories as they are constructed in the Fairface dataset.) in order to assess quality of performance across different demographics. We found accuracy >96% across all races for gender classification with ‘Middle Eastern’ having the highest accuracy (98.4%) and ‘White’ having the lowest (96.5%). Additionally, CLIP averaged ~93% for racial classification and ~63% for age classification. Our use of evaluations to test for gender, race and age classification as well as denigration harms is simply to evaluate performance of the model across people and surface potential risks and not to demonstrate an endorsement/enthusiasm for such tasks."});
CREATE (mcro_TheBlokephi2GGUF:Model {id: "mcro:TheBlokephi2GGUF"});
CREATE (mcro_TheBlokephi2GGUF_Description:IAO_0000310 {id: "mcro:TheBlokephi2GGUF-Description", hasTextValue: "This repo contains GGUF format model files for [Microsoft's Phi 2](https://huggingface.co/microsoft/phi-2)."});
CREATE (mcro_TheBlokephi2GGUF_License:LicenseInformationSection {id: "mcro:TheBlokephi2GGUF-License", hasTextValue: "The model is licensed under the [microsoft-research-license](https://huggingface.co/microsoft/phi-2/resolve/main/LICENSE)."});
CREATE (mcro_TheBlokephi2GGUF_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:TheBlokephi2GGUF-ModelArchitecture", hasTextValue: "Phi-2 is a Transformer with **2.7 billion** parameters."});
CREATE (mcro_TheBlokephi2GGUF_IntendedUseCase:UseCaseInformationSection {id: "mcro:TheBlokephi2GGUF-IntendedUseCase", hasTextValue: "Phi-2 is intended for research purposes only."});
CREATE (mcro_TheBlokephi2GGUF_Limitation:LimitationInformationSection {id: "mcro:TheBlokephi2GGUF-Limitation", hasTextValue: "* Generate Inaccurate Code and Facts: The model may produce incorrect code snippets and statements. Users should treat these outputs as suggestions or starting points, not as definitive or accurate solutions.

* Limited Scope for code: Majority of Phi-2 training data is based in Python and use common packages such as \"typing, math, random, collections, datetime, itertools\". If the model generates Python scripts that utilize other packages or scripts in other languages, we strongly recommend users manually verify all API uses.

* Unreliable Responses to Instruction: The model has not undergone instruction fine-tuning. As a result, it may struggle or fail to adhere to intricate or nuanced instructions provided by users.

* Language Limitations: The model is primarily designed to understand standard English. Informal English, slang, or any other languages might pose challenges to its comprehension, leading to potential misinterpretations or errors in response.

* Potential Societal Biases: Phi-2 is not entirely free from societal biases despite efforts in assuring trainig data safety. There's a possibility it may generate content that mirrors these societal biases, particularly if prompted or instructed to do so. We urge users to be aware of this and to exercise caution and critical thinking when interpreting model outputs.

* Toxicity: Despite being trained with carefully selected data, the model can still produce harmful content if explicitly prompted or instructed to do so. We chose to release the model for research purposes only -- We hope to help the open-source community develop the most effective ways to reduce the toxicity of a model directly after pretraining.

* Verbosity: Phi-2 being a base model often produces irrelevant or extra text and responses following its first answer to user prompts within a single turn. This is due to its training dataset being primarily textbooks, which results in textbook-like responses."});
CREATE (mcro_TheBlokephi2GGUF_Dataset:DatasetInformationSection {id: "mcro:TheBlokephi2GGUF-Dataset", hasTextValue: "Dataset size: 250B tokens, combination of NLP synthetic data created by AOAI GPT-3.5 and filtered web data from Falcon RefinedWeb and SlimPajama, which was assessed by AOAI GPT-4."});
CREATE (mcro_TheBlokephi2GGUF_Citation:CitationInformationSection {id: "mcro:TheBlokephi2GGUF-Citation", hasTextValue: "Microsoft's Phi 2"});
CREATE (mcro_chronos_t5_small:Model {id: "mcro:chronos-t5-small"});
CREATE (mcro_chronos_t5_small_architecture:ModelArchitectureInformationSection {id: "mcro:chronos-t5-small-architecture", hasTextValue: "based on the [T5 architecture](https://arxiv.org/abs/1910.10683)"});
CREATE (mcro_chronos_t5_small_citation:CitationInformationSection {id: "mcro:chronos-t5-small-citation", hasTextValue: "@articleansari2024chronos,
    title=Chronos: Learning the Language of Time Series,
    author=Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Caner and Zhang, Xiyuan, and Mercado, Pedro and Shen, Huibin and Shchur, Oleksandr and Rangapuram, Syama Syndar and Pineda Arango, Sebastian and Kapoor, Shubham and Zschiegner, Jasper and Maddix, Danielle C. and Mahoney, Michael W. and Torkkola, Kari and Gordon Wilson, Andrew and Bohlke-Schneider, Michael and Wang, Yuyang,
    journal=Transactions on Machine Learning Research,
    issn=2835-8856,
    year=2024,
    url=https://openreview.net/forum?id=gerNCVqqtR"});
CREATE (mcro_chronos_t5_small_license:LicenseInformationSection {id: "mcro:chronos-t5-small-license", hasTextValue: "Apache-2.0 License"});
CREATE (mcro_chronos_t5_small_usecase:UseCaseInformationSection {id: "mcro:chronos-t5-small-usecase", hasTextValue: "pretrained time series forecasting models"});
CREATE (mcro_chronos_t5_small_dataset:DatasetInformationSection {id: "mcro:chronos-t5-small-dataset", hasTextValue: "large corpus of publicly available time series data, as well as synthetic data generated using Gaussian processes."});
CREATE (mcro_robertalargemodel:Model {id: "mcro:robertalargemodel"});
CREATE (mcro_robertalargemodel_Architecture:ModelArchitectureInformationSection {id: "mcro:robertalargemodel-Architecture", hasTextValue: "transformers model"});
CREATE (mcro_robertalargemodel_UseCase:UseCaseInformationSection {id: "mcro:robertalargemodel-UseCase", hasTextValue: "masked language modeling"});
CREATE (mcro_robertalargemodel_TrainingData:TrainingDataInformationSection {id: "mcro:robertalargemodel-TrainingData", hasTextValue: "Stories"});
CREATE (mcro_ESMFold:Model {id: "mcro:ESMFold"});
CREATE (mcro_ESMFold_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:ESMFold-ModelArchitecture", hasTextValue: "ESMFold is a state-of-the-art end-to-end protein folding model based on an ESM-2 backbone."});
CREATE (mcro_ESMFold_Citation:CitationInformationSection {id: "mcro:ESMFold-Citation", hasTextValue: "For details on the model architecture and training, please refer to the accompanying paper"});
CREATE (mcro_YOLOv8DetectionModel:Model {id: "mcro:YOLOv8DetectionModel"});
CREATE (mcro_YOLOv8DetectionModel_DatasetInformationSection:DatasetInformationSection {id: "mcro:YOLOv8DetectionModel-DatasetInformationSection"});
CREATE (mcro_YOLOv8DetectionModel_UseCaseInformationSection:UseCaseInformationSection {id: "mcro:YOLOv8DetectionModel-UseCaseInformationSection"});
CREATE (mcro_YOLOv8DetectionModel_ModelArchitectureInformationSection:ModelArchitectureInformationSection {id: "mcro:YOLOv8DetectionModel-ModelArchitectureInformationSection"});
CREATE (mcro_allmpnetbasev2:Model {id: "mcro:allmpnetbasev2"});
CREATE (mcro_allmpnetbasev2_UseCaseInformationSection:UseCaseInformationSection {id: "mcro:allmpnetbasev2-UseCaseInformationSection", hasTextValue: "Our model is intented to be used as a sentence and short paragraph encoder. Given an input text, it ouptuts a vector which captures 
the semantic information. The sentence vector may be used for information retrieval, clustering or sentence similarity tasks.

By default, input text longer than 384 word pieces is truncated."});
CREATE (mcro_allmpnetbasev2_TrainingDataInformationSection:TrainingDataInformationSection {id: "mcro:allmpnetbasev2-TrainingDataInformationSection", hasTextValue: "We use the concatenation from multiple datasets to fine-tune our model. The total number of sentence pairs is above 1 billion sentences.
We sampled each dataset given a weighted probability which configuration is detailed in the `data_config.json` file."});
CREATE (mcro_allmpnetbasev2_ModelArchitectureInformationSection:ModelArchitectureInformationSection {id: "mcro:allmpnetbasev2-ModelArchitectureInformationSection", hasTextValue: "We used the pretrained [`microsoft/mpnet-base`](https://huggingface.co/microsoft/mpnet-base) model and fine-tuned in on a 
1B sentence pairs dataset."});
CREATE (mcro_electramodel:Model {id: "mcro:electramodel"});
CREATE (mcro_electramodel_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:electramodel-ModelArchitecture", hasTextValue: "transformer networks"});
CREATE (mcro_electramodel_UseCase:UseCaseInformationSection {id: "mcro:electramodel-UseCase", hasTextValue: "sequence tagging tasks"});
CREATE (mcro_electramodel_Citation:CitationInformationSection {id: "mcro:electramodel-Citation", hasTextValue: "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators"});
CREATE (mcro_pyannotewespeakervoxcelebresnet34LM:Model {id: "mcro:pyannotewespeakervoxcelebresnet34LM"});
CREATE (mcro_pyannotewespeakervoxcelebresnet34LM_License:LicenseInformationSection {id: "mcro:pyannotewespeakervoxcelebresnet34LM-License", hasTextValue: "According to [this page](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md):

> The pretrained model in WeNet follows the license of it's corresponding dataset. For example, the pretrained model on VoxCeleb follows Creative Commons Attribution 4.0 International License., since it is used as license of the VoxCeleb dataset, see https://mm.kaist.ac.kr/datasets/voxceleb/."});
CREATE (mcro_pyannotewespeakervoxcelebresnet34LM_Citation1:CitationInformationSection {id: "mcro:pyannotewespeakervoxcelebresnet34LM-Citation1", hasTextValue: "@inproceedingsWang2023,
  title=Wespeaker: A research and production oriented speaker embedding learning toolkit,
  author=Wang, Hongji and Liang, Chengdong and Wang, Shuai and Chen, Zhengyang and Zhang, Binbin and Xiang, Xu and Deng, Yanlei and Qian, Yanmin,
  booktitle=ICASSP 2023, IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),
  pages=1--5,
  year=2023,
  organization=IEEE"});
CREATE (mcro_pyannotewespeakervoxcelebresnet34LM_Citation2:CitationInformationSection {id: "mcro:pyannotewespeakervoxcelebresnet34LM-Citation2", hasTextValue: "@inproceedingsBredin23,
  author=Hervé Bredin,
  title=pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe,
  year=2023,
  booktitle=Proc. INTERSPEECH 2023,
  pages=1983--1987,
  doi=10.21437/Interspeech.2023-105"});
CREATE (mcro_pyannotewespeakervoxcelebresnet34LM_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:pyannotewespeakervoxcelebresnet34LM-ModelArchitecture", hasTextValue: "wrapper around [WeSpeaker](https://github.com/wenet-e2e/wespeaker) `wespeaker-voxceleb-resnet34-LM` pretrained speaker embedding model, for use in `pyannote.audio`."});
CREATE (mcro_resnet50a1in1k:Model {id: "mcro:resnet50a1in1k"});
CREATE (mcro_resnet50a1in1k_ModelDetail:ModelDetailSection {id: "mcro:resnet50a1in1k-ModelDetail"});
CREATE (mcro_resnet50a1in1k_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:resnet50a1in1k-ModelArchitecture", hasTextValue: "ResNet-B image classification model"});
CREATE (mcro_resnet50a1in1k_Citation:CitationInformationSection {id: "mcro:resnet50a1in1k-Citation", hasTextValue: "@articleHe2015,
  author = Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun,
  title = Deep Residual Learning for Image Recognition,
  journal = arXiv preprint arXiv:1512.03385,
  year = 2015"});
CREATE (mcro_resnet50a1in1k_UseCase:UseCaseInformationSection {id: "mcro:resnet50a1in1k-UseCase", hasTextValue: "Image Embeddings"});
CREATE (mcro_resnet50a1in1k_Dataset:DatasetInformationSection {id: "mcro:resnet50a1in1k-Dataset", hasTextValue: "ImageNet-1k"});
CREATE (mcro_llama318BInstructGGUF:Model {id: "mcro:llama318BInstructGGUF"});
CREATE (mcro_llama318BInstructGGUF_ModelDetail:ModelDetailSection {id: "mcro:llama318BInstructGGUF-ModelDetail"});
CREATE (mcro_llama318BInstructGGUF_License:LicenseInformationSection {id: "mcro:llama318BInstructGGUF-License"});
CREATE (mcro_llama318BInstructGGUF_Architecture:ModelArchitectureInformationSection {id: "mcro:llama318BInstructGGUF-Architecture"});
CREATE (mcro_llama318BInstructGGUF_IntendedUse:UseCaseInformationSection {id: "mcro:llama318BInstructGGUF-IntendedUse"});
CREATE (mcro_llama318BInstructGGUF_Dataset:DatasetInformationSection {id: "mcro:llama318BInstructGGUF-Dataset"});
CREATE (mcro_clip_CitationInformationSection:CitationInformationSection {id: "mcro:clip-CitationInformationSection", hasTextValue: "CLIP Paper"});
CREATE (mcro_clip_ModelDetailSection:ModelDetailSection {id: "mcro:clip-ModelDetailSection", hasTextValue: "The CLIP model was developed by researchers at OpenAI to learn about what contributes to robustness in computer vision tasks. The model was also developed to test the ability of models to generalize to arbitrary image classification tasks in a zero-shot manner. It was not developed for general model deployment - to deploy models like CLIP, researchers will first need to carefully study their capabilities in relation to the specific context they’re being deployed within."});
CREATE (mcro_clip_VersionInformationSection:VersionInformationSection {id: "mcro:clip-VersionInformationSection", hasTextValue: "January 2021"});
CREATE (mcro_clip_ModelArchitectureInformationSection:ModelArchitectureInformationSection {id: "mcro:clip-ModelArchitectureInformationSection", hasTextValue: "The base model uses a ViT-B/16 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder."});
CREATE (mcro_clip_ReferenceInformationSection:ReferenceInformationSection {id: "mcro:clip-ReferenceInformationSection", hasTextValue: "- [Blog Post](https://openai.com/blog/clip/)
- [CLIP Paper](https://arxiv.org/abs/2103.00020)"});
CREATE (mcro_clip_UseCaseInformationSection:UseCaseInformationSection {id: "mcro:clip-UseCaseInformationSection", hasTextValue: "The model is intended as a research output for research communities. We hope that this model will enable researchers to better understand and explore zero-shot, arbitrary image classification. We also hope it can be used for interdisciplinary studies of the potential impact of such models - the CLIP paper includes a discussion of potential downstream impacts to provide an example for this sort of analysis."});
CREATE (mcro_clip_UserInformationSection:UserInformationSection {id: "mcro:clip-UserInformationSection", hasTextValue: "The primary intended users of these models are AI researchers.

We primarily imagine the model will be used by researchers to better understand robustness, generalization, and other capabilities, biases, and constraints of computer vision models."});
CREATE (mcro_clip_ConsiderationInformationSection:ConsiderationInformationSection {id: "mcro:clip-ConsiderationInformationSection", hasTextValue: "**Any** deployed use case of the model - whether commercial or not - is currently out of scope. Non-deployed use cases such as image search in a constrained environment, are also not recommended unless there is thorough in-domain testing of the model with a specific, fixed class taxonomy. This is because our safety assessment demonstrated a high need for task specific testing especially given the variability of CLIP’s performance with different class taxonomies. This makes untested and unconstrained deployment of the model in any use case currently potentially harmful. 

Certain use cases which would fall under the domain of surveillance and facial recognition are always out-of-scope regardless of performance of the model. This is because the use of artificial intelligence for tasks such as these can be premature currently given the lack of testing norms and checks to ensure its fair use.

Since the model has not been purposefully trained in or evaluated on any languages other than English, its use should be limited to English language use cases."});
CREATE (mcro_clip_DatasetInformationSection:DatasetInformationSection {id: "mcro:clip-DatasetInformationSection", hasTextValue: "The model was trained on publicly available image-caption data."});
CREATE (mcro_clip_ConsiderationInformationSection2:ConsiderationInformationSection {id: "mcro:clip-ConsiderationInformationSection2", hasTextValue: "Our goal with building this dataset was to test out robustness and generalizability in computer vision tasks. As a result, the focus was on gathering large quantities of data from different publicly-available internet data sources. The data was gathered in a mostly non-interventionist manner. However, we only crawled websites that had policies against excessively violent and adult images and allowed us to filter out such content. We do not intend for this dataset to be used as the basis for any commercial or deployed model and will not be releasing the dataset."});
CREATE (mcro_clip_QuantativeAnalysisSection:QuantativeAnalysisSection {id: "mcro:clip-QuantativeAnalysisSection", hasTextValue: "We have evaluated the performance of CLIP on a wide range of benchmarks across a variety of computer vision datasets such as OCR to texture recognition to fine-grained classification. The paper describes model performance on the following datasets:

- Food101
- CIFAR10   
- CIFAR100   
- Birdsnap
- SUN397
- Stanford Cars
- FGVC Aircraft
- VOC2007
- DTD
- Oxford-IIIT Pet dataset
- Caltech101
- Flowers102
- MNIST   
- SVHN 
- IIIT5K   
- Hateful Memes   
- SST-2
- UCF101
- Kinetics700
- Country211
- CLEVR Counting
- KITTI Distance
- STL-10
- RareAct
- Flickr30
- MSCOCO
- ImageNet
- ImageNet-A
- ImageNet-R
- ImageNet Sketch
- ObjectNet (ImageNet Overlap)
- Youtube-BB
- ImageNet-Vid"});
CREATE (mcro_clip_LimitationInformationSection:LimitationInformationSection {id: "mcro:clip-LimitationInformationSection", hasTextValue: "CLIP currently struggles with respect to certain tasks such as fine grained classification and counting objects."});
CREATE (mcro_clip_RiskInformationSection:RiskInformationSection {id: "mcro:clip-RiskInformationSection", hasTextValue: "We find that the performance of CLIP - and the specific biases it exhibits - can depend significantly on class design and the choices one makes for categories to include and exclude. We tested the risk of certain kinds of denigration with CLIP by classifying images of people from [Fairface](https://arxiv.org/abs/1908.04913) into crime-related and non-human animal categories. We found significant disparities with respect to race and gender. Additionally, we found that these disparities could shift based on how the classes were constructed. (Details captured in the Broader Impacts Section in the paper).

We also tested the performance of CLIP on gender, race and age classification using the Fairface dataset (We default to using race categories as they are constructed in the Fairface dataset.) in order to assess quality of performance across different demographics. We found accuracy >96% across all races for gender classification with ‘Middle Eastern’ having the highest accuracy (98.4%) and ‘White’ having the lowest (96.5%). Additionally, CLIP averaged ~93% for racial classification and ~63% for age classification. Our use of evaluations to test for gender, race and age classification as well as denigration harms is simply to evaluate performance of the model across people and surface potential risks and not to demonstrate an endorsement/enthusiasm for such tasks."});
CREATE (mcro_pyannotesegmentation30:Model {id: "mcro:pyannotesegmentation30"});
CREATE (mcro_pyannotesegmentation30_UseCase:UseCaseInformationSection {id: "mcro:pyannotesegmentation30-UseCase", hasTextValue: "speaker segmentation"});
CREATE (mcro_pyannotesegmentation30_InputFormat:FormatInformationSection {id: "mcro:pyannotesegmentation30-InputFormat", hasTextValue: "10 seconds of mono audio sampled at 16kHz"});
CREATE (mcro_pyannotesegmentation30_OutputFormat:FormatInformationSection {id: "mcro:pyannotesegmentation30-OutputFormat", hasTextValue: "(num_frames, num_classes) matrix where the 7 classes are _non-speech_, _speaker #1_, _speaker #2_, _speaker #3_, _speakers #1 and #2_, _speakers #1 and #3_, and _speakers #2 and #3_"});
CREATE (mcro_pyannotesegmentation30_Dataset:DatasetInformationSection {id: "mcro:pyannotesegmentation30-Dataset", hasTextValue: "AISHELL, AliMeeting, AMI, AVA-AVD, DIHARD, Ego4D, MSDWild, REPERE, and VoxConverse"});
CREATE (mcro_pyannotesegmentation30_Citation1:CitationInformationSection {id: "mcro:pyannotesegmentation30-Citation1", hasTextValue: "@inproceedingsPlaquet23,
  author=Alexis Plaquet and Hervé Bredin,
  title=Powerset multi-class cross entropy loss for neural speaker diarization,
  year=2023,
  booktitle=Proc. INTERSPEECH 2023,"});
CREATE (mcro_pyannotesegmentation30_Citation2:CitationInformationSection {id: "mcro:pyannotesegmentation30-Citation2", hasTextValue: "@inproceedingsBredin23,
  author=Hervé Bredin,
  title=pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe,
  year=2023,
  booktitle=Proc. INTERSPEECH 2023,"});
CREATE (mcro_gpt2:Model {id: "mcro:gpt2"});
CREATE (mcro_gpt2_ModelDetail:ModelDetailSection {id: "mcro:gpt2-ModelDetail"});
CREATE (mcro_gpt2_Citation:CitationInformationSection {id: "mcro:gpt2-Citation", hasTextValue: "@articleradford2019language,
  title=Language Models are Unsupervised Multitask Learners,
  author=Radford, Alec and Wu, Jeff and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya,
  year=2019"});
CREATE (mcro_gpt2_UseCase:UseCaseInformationSection {id: "mcro:gpt2-UseCase"});
CREATE (mcro_gpt2_Consideration:ConsiderationInformationSection {id: "mcro:gpt2-Consideration"});
CREATE (mcro_gpt2_TrainingData:TrainingDataInformationSection {id: "mcro:gpt2-TrainingData"});
CREATE (mcro_gpt2_Evaluation:QuantativeAnalysisSection {id: "mcro:gpt2-Evaluation"});
CREATE (mcro_distilbertbasemodeluncased:Model {id: "mcro:distilbertbasemodeluncased"});
CREATE (mcro_distilbertbasemodeluncased_ModelDetailSection:ModelDetailSection {id: "mcro:distilbertbasemodeluncased-ModelDetailSection"});
CREATE (mcro_distilbertbasemodeluncased_CitationInformationSection:CitationInformationSection {id: "mcro:distilbertbasemodeluncased-CitationInformationSection", hasTextValue: "@articleSanh2019DistilBERTAD,
  title=DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter,
  author=Victor Sanh and Lysandre Debut and Julien Chaumond and Thomas Wolf,
  journal=ArXiv,
  year=2019,
  volume=abs/1910.01108"});
CREATE (mcro_distilbertbasemodeluncased_LicenseInformationSection:LicenseInformationSection {id: "mcro:distilbertbasemodeluncased-LicenseInformationSection"});
CREATE (mcro_distilbertbasemodeluncased_UseCaseInformationSection:UseCaseInformationSection {id: "mcro:distilbertbasemodeluncased-UseCaseInformationSection"});
CREATE (mcro_distilbertbasemodeluncased_TrainingDataInformationSection:TrainingDataInformationSection {id: "mcro:distilbertbasemodeluncased-TrainingDataInformationSection", hasTextValue: "DistilBERT pretrained on the same data as BERT, which is [BookCorpus](https://yknzhu.wixsite.com/mbweb), a dataset
consisting of 11,038 unpublished books and [English Wikipedia](https://en.wikipedia.org/wiki/English_Wikipedia)
(excluding lists, tables and headers)."});
CREATE (mcro_distilbertbasemodeluncased_LimitationInformationSection:LimitationInformationSection {id: "mcro:distilbertbasemodeluncased-LimitationInformationSection"});
CREATE (mcro_clipsegModel:Model {id: "mcro:clipsegModel"});
CREATE (mcro_clipsegModel_UseCase:UseCaseInformationSection {id: "mcro:clipsegModel-UseCase", hasTextValue: "This model is intended for zero-shot and one-shot image segmentation."});
CREATE (mcro_clipsegModel_Citation:CitationInformationSection {id: "mcro:clipsegModel-Citation", hasTextValue: "It was introduced in the paper [Image Segmentation Using Text and Image Prompts](https://arxiv.org/abs/2112.10003) by Lüddecke et al. and first released in [this repository](https://github.com/timojl/clipseg)."});
CREATE (mcro_pyannotespeakerdiarization31:Model {id: "mcro:pyannotespeakerdiarization31"});
CREATE (mcro_pyannotespeakerdiarization31_Citation:CitationInformationSection {id: "mcro:pyannotespeakerdiarization31-Citation", hasTextValue: "@inproceedingsPlaquet23,
  author=Alexis Plaquet and Hervé Bredin,
  title=Powerset multi-class cross entropy loss for neural speaker diarization,
  year=2023,
  booktitle=Proc. INTERSPEECH 2023,"});
CREATE (mcro_pyannotespeakerdiarization31_Citation2:CitationInformationSection {id: "mcro:pyannotespeakerdiarization31-Citation2", hasTextValue: "@inproceedingsBredin23,
  author=Hervé Bredin,
  title=pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe,
  year=2023,
  booktitle=Proc. INTERSPEECH 2023,"});
CREATE (mcro_XLMROBERTaModel:Model {id: "mcro:XLMROBERTaModel"});
CREATE (mcro_XLMROBERTaModel_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:XLMROBERTaModel-ModelArchitecture", hasTextValue: "transformers"});
CREATE (mcro_XLMROBERTaModel_IntendedUseCase:UseCaseInformationSection {id: "mcro:XLMROBERTaModel-IntendedUseCase", hasTextValue: "masked language modeling"});
CREATE (mcro_XLMROBERTaModel_Citation:CitationInformationSection {id: "mcro:XLMROBERTaModel-Citation", hasTextValue: "Alexis Conneau and
               Kartikay Khandelwal and
               Naman Goyal and
               Vishrav Chaudhary and
               Guillaume Wenzek and
               Francisco Guzm'an and
               Edouard Grave and
               Myle Ott and
               Luke Zettlemoyer and
               Veselin Stoyanov"});
CREATE (mcro_RoBERTa_base_model:Model {id: "mcro:RoBERTa_base_model"});
CREATE (mcro_RoBERTa_base_model_CitationInformationSection:CitationInformationSection {id: "mcro:RoBERTa_base_model-CitationInformationSection", hasTextValue: "@articleDBLP:journals/corr/abs-1907-11692,
  author    = Yinhan Liu and
               Myle Ott and
               Naman Goyal and
               Jingfei Du and
               Mandar Joshi and
               Danqi Chen and
               Omer Levy and
               Mike Lewis and
               Luke Zettlemoyer and
               Veselin Stoyanov,
  title     = RoBERTa: A Robustly Optimized BERT Pretraining Approach,
  journal   = CoRR,
  volume    = abs/1907.11692,
  year      = 2019,
  url       = http://arxiv.org/abs/1907.11692,
  archivePrefix = arXiv,
  eprint    = 1907.11692,
  timestamp = Thu, 01 Aug 2019 08:59:33 +0200,
  biburl    = https://dblp.org/rec/journals/corr/abs-1907-11692.bib,
  bibsource = dblp computer science bibliography, https://dblp.org"});
CREATE (mcro_RoBERTa_base_model_ModelArchitectureInformationSection:ModelArchitectureInformationSection {id: "mcro:RoBERTa_base_model-ModelArchitectureInformationSection", hasTextValue: "RoBERTa is a transformers model pretrained on a large corpus of English data in a self-supervised fashion"});
CREATE (mcro_RoBERTa_base_model_UseCaseInformationSection:UseCaseInformationSection {id: "mcro:RoBERTa_base_model-UseCaseInformationSection", hasTextValue: "You can use the raw model for masked language modeling, but it's mostly intended to be fine-tuned on a downstream task."});
CREATE (mcro_RoBERTa_base_model_DatasetInformationSection:DatasetInformationSection {id: "mcro:RoBERTa_base_model-DatasetInformationSection", hasTextValue: "The RoBERTa model was pretrained on the reunion of five datasets:
- [BookCorpus](https://yknzhu.wixsite.com/mbweb), a dataset consisting of 11,038 unpublished books;
- [English Wikipedia](https://en.wikipedia.org/wiki/English_Wikipedia) (excluding lists, tables and headers) ;
- [CC-News](https://commoncrawl.org/2016/10/news-dataset-available/), a dataset containing 63 millions English news
  articles crawled between September 2016 and February 2019.
- [OpenWebText](https://github.com/jcpeterson/openwebtext), an opensource recreation of the WebText dataset used to
  train GPT-2,
- [Stories](https://arxiv.org/abs/1806.02847) a dataset containing a subset of CommonCrawl data filtered to match the
  story-like style of Winograd schemas.

Together these datasets weigh 160GB of text."});
CREATE (mcro_sentencetransformersparaphrasemultilingualMiniLML12v2:Model {id: "mcro:sentencetransformersparaphrasemultilingualMiniLML12v2"});
CREATE (mcro_sentencetransformersparaphrasemultilingualMiniLML12v2_UseCase:UseCaseInformationSection {id: "mcro:sentencetransformersparaphrasemultilingualMiniLML12v2-UseCase", hasTextValue: "clustering or semantic search"});
CREATE (mcro_sentencetransformersparaphrasemultilingualMiniLML12v2_Architecture:ModelArchitectureInformationSection {id: "mcro:sentencetransformersparaphrasemultilingualMiniLML12v2-Architecture", hasTextValue: "SentenceTransformer(
  (0): Transformer('max_seq_length': 128, 'do_lower_case': False) with Transformer model: BertModel 
  (1): Pooling('word_embedding_dimension': 384, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False)
)"});
CREATE (mcro_sentencetransformersparaphrasemultilingualMiniLML12v2_Citation:CitationInformationSection {id: "mcro:sentencetransformersparaphrasemultilingualMiniLML12v2-Citation", hasTextValue: "@inproceedingsreimers-2019-sentence-bert,
    title = \"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks\",
    author = \"Reimers, Nils and Gurevych, Iryna\",
    booktitle = \"Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing\",
    month = \"11\",
    year = \"2019\",
    publisher = \"Association for Computational Linguistics\",
    url = \"http://arxiv.org/abs/1908.10084\","});
CREATE (mcro_chronosboltbase:Model {id: "mcro:chronosboltbase"});
CREATE (mcro_chronosboltbase_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:chronosboltbase-ModelArchitecture", hasTextValue: "T5 encoder-decoder architecture"});
CREATE (mcro_chronosboltbase_Citation:CitationInformationSection {id: "mcro:chronosboltbase-Citation", hasTextValue: "@articleansari2024chronos,
    title=Chronos: Learning the Language of Time Series,
    author=Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Caner and Zhang, Xiyuan, and Mercado, Pedro and Shen, Huibin and Shchur, Oleksandr and Rangapuram, Syama Syndar and Pineda Arango, Sebastian and Kapoor, Shubham and Zschiegner, Jasper and Maddix, Danielle C. and Mahoney, Michael W. and Torkkola, Kari and Gordon Wilson, Andrew and Bohlke-Schneider, Michael and Wang, Yuyang,
    journal=Transactions on Machine Learning Research,
    issn=2835-8856,
    year=2024,
    url=https://openreview.net/forum?id=gerNCVqqtR"});
CREATE (mcro_chronosboltbase_License:LicenseInformationSection {id: "mcro:chronosboltbase-License", hasTextValue: "Apache-2.0 License"});
CREATE (mcro_chronosboltbase_UseCase:UseCaseInformationSection {id: "mcro:chronosboltbase-UseCase", hasTextValue: "zero-shot forecasting"});
CREATE (mcro_sentencetransformersusecmlmmultilingual:Model {id: "mcro:sentencetransformersusecmlmmultilingual", hasTextValue: "This is a pytorch version of the [universal-sentence-encoder-cmlm/multilingual-base-br](https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base-br/1) model. It can be used to map 109 languages to a shared vector space. As the model is based [LaBSE](https://huggingface.co/sentence-transformers/LaBSE), it perform quite comparable on downstream tasks."});
CREATE (mcro_sentencetransformersusecmlmmultilingual_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:sentencetransformersusecmlmmultilingual-ModelArchitecture", hasTextValue: "SentenceTransformer(
  (0): Transformer('max_seq_length': 256, 'do_lower_case': False) with Transformer model: BertModel 
  (1): Pooling('word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False)
  (2): Normalize()
)"});
CREATE (mcro_sentencetransformersusecmlmmultilingual_Citation:CitationInformationSection {id: "mcro:sentencetransformersusecmlmmultilingual-Citation", hasTextValue: "Have a look at [universal-sentence-encoder-cmlm/multilingual-base-br](https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base-br/1) for the respective publication that describes this model."});
CREATE (mcro_whisperlargev3:Model {id: "mcro:whisperlargev3"});
CREATE (mcro_whisperlargev3_Citation:CitationInformationSection {id: "mcro:whisperlargev3-Citation", hasTextValue: "@miscradford2022whisper,
  doi = 10.48550/ARXIV.2212.04356,
  url = https://arxiv.org/abs/2212.04356,
  author = Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya,
  title = Robust Speech Recognition via Large-Scale Weak Supervision,
  publisher = arXiv,
  year = 2022,
  copyright = arXiv.org perpetual, non-exclusive license"});
CREATE (mcro_whisperlargev3_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:whisperlargev3-ModelArchitecture", hasTextValue: "Whisper is a Transformer based encoder-decoder model, also referred to as a _sequence-to-sequence_ model"});
CREATE (mcro_whisperlargev3_TrainingData:TrainingDataInformationSection {id: "mcro:whisperlargev3-TrainingData", hasTextValue: "The large-v3 checkpoint is trained on 1 million hours of weakly labeled audio and 4 million hours of pseudo-labeled audio collected using Whisper large-v2."});
CREATE (mcro_whisperlargev3_IntendedUseCase:UseCaseInformationSection {id: "mcro:whisperlargev3-IntendedUseCase", hasTextValue: "The models are primarily trained and evaluated on ASR and speech translation to English tasks."});
CREATE (mcro_whisperlargev3turbo:Model {id: "mcro:whisperlargev3turbo"});
CREATE (mcro_whisperlargev3turbo_Citation:CitationInformationSection {id: "mcro:whisperlargev3turbo-Citation", hasTextValue: "@miscradford2022whisper,
  doi = 10.48550/ARXIV.2212.04356,
  url = https://arxiv.org/abs/2212.04356,
  author = Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya,
  title = Robust Speech Recognition via Large-Scale Weak Supervision,
  publisher = arXiv,
  year = 2022,
  copyright = arXiv.org perpetual, non-exclusive license"});
CREATE (mcro_whisperlargev3turbo_Architecture:ModelArchitectureInformationSection {id: "mcro:whisperlargev3turbo-Architecture", hasTextValue: "Whisper is a Transformer based encoder-decoder model, also referred to as a _sequence-to-sequence_ model."});
CREATE (mcro_whisperlargev3turbo_IntendedUseCase:UseCaseInformationSection {id: "mcro:whisperlargev3turbo-IntendedUseCase", hasTextValue: "The models are primarily trained and evaluated on ASR and speech translation to English tasks."});
CREATE (mcro_whisperlargev3turbo_Dataset:DatasetInformationSection {id: "mcro:whisperlargev3turbo-Dataset", hasTextValue: "Trained on >5M hours of labeled data, Whisper demonstrates a strong ability to generalise to many datasets and domains in a zero-shot setting."});
CREATE (mcro_bertmultilingualbasemodelcased:Model {id: "mcro:bertmultilingualbasemodelcased"});
CREATE (mcro_bertmultilingualbasemodelcased_ModelDetail:ModelDetailSection {id: "mcro:bertmultilingualbasemodelcased-ModelDetail"});
CREATE (mcro_bertmultilingualbasemodelcased_Citation:CitationInformationSection {id: "mcro:bertmultilingualbasemodelcased-Citation"});
CREATE (mcro_bertmultilingualbasemodelcased_UseCase:UseCaseInformationSection {id: "mcro:bertmultilingualbasemodelcased-UseCase"});
CREATE (mcro_bertmultilingualbasemodelcased_TrainingData:DatasetInformationSection {id: "mcro:bertmultilingualbasemodelcased-TrainingData"});
CREATE (mcro_bertmultilingualbasemodelcased_TrainingProcedure:ModelParameterSection {id: "mcro:bertmultilingualbasemodelcased-TrainingProcedure"});
CREATE (mcro_bertmultilingualbasemodelcased_Citation2:CitationInformationSection {id: "mcro:bertmultilingualbasemodelcased-Citation2", hasTextValue: "@articleDBLP:journals/corr/abs-1810-04805,
  author    = Jacob Devlin and
               Ming-Wei Chang and
               Kenton Lee and
               Kristina Toutanova,
  title     = BERT: Pre-training of Deep Bidirectional Transformers for Language
               Understanding,
  journal   = CoRR,
  volume    = abs/1810.04805,
  year      = 2018,
  url       = http://arxiv.org/abs/1810.04805,
  archivePrefix = arXiv,
  eprint    = 1810.04805,
  timestamp = Tue, 30 Oct 2018 20:39:56 +0100,
  biburl    = https://dblp.org/rec/journals/corr/abs-1810-04805.bib,
  bibsource = dblp computer science bibliography, https://dblp.org"});
CREATE (mcro_vit_face_expression:Model {id: "mcro:vit-face-expression"});
CREATE (mcro_vit_face_expression_Dataset:DatasetInformationSection {id: "mcro:vit-face-expression-Dataset", hasTextValue: "FER2013"});
CREATE (mcro_vit_face_expression_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:vit-face-expression-ModelArchitecture", hasTextValue: "Vision Transformer (ViT)"});
CREATE (mcro_vit_face_expression_Limitation:LimitationInformationSection {id: "mcro:vit-face-expression-Limitation", hasTextValue: "Generalization"});
CREATE (mcro_opt_125m:Model {id: "mcro:opt-125m"});
CREATE (mcro_opt_125m_ModelDetail:ModelDetailSection {id: "mcro:opt-125m-ModelDetail"});
CREATE (mcro_opt_125m_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:opt-125m-ModelArchitecture", hasTextValue: "decoder-only"});
CREATE (mcro_opt_125m_Citation:CitationInformationSection {id: "mcro:opt-125m-Citation"});
CREATE (mcro_opt_125m_License:LicenseInformationSection {id: "mcro:opt-125m-License"});
CREATE (mcro_opt_125m_IntendedUseCase:UseCaseInformationSection {id: "mcro:opt-125m-IntendedUseCase"});
CREATE (mcro_opt_125m_Dataset:DatasetInformationSection {id: "mcro:opt-125m-Dataset", hasTextValue: "CCNewsV2"});
CREATE (mcro_opt_125m_Consideration:ConsiderationInformationSection {id: "mcro:opt-125m-Consideration", hasTextValue: "bias"});
CREATE (mcro_opt_125m_Training:TrainingDataInformationSection {id: "mcro:opt-125m-Training"});
CREATE (mcro_opt_125m_ModelParameter:ModelParameterSection {id: "mcro:opt-125m-ModelParameter"});
CREATE (mcro_siglipso400mpatch14384:Model {id: "mcro:siglipso400mpatch14384"});
CREATE (mcro_siglipso400mpatch14384_ModelDetail:ModelDetailSection {id: "mcro:siglipso400mpatch14384-ModelDetail"});
CREATE (mcro_siglipso400mpatch14384_Architecture:ModelArchitectureInformationSection {id: "mcro:siglipso400mpatch14384-Architecture", hasTextValue: "SoViT-400m architecture"});
CREATE (mcro_siglipso400mpatch14384_Citation:CitationInformationSection {id: "mcro:siglipso400mpatch14384-Citation", hasTextValue: "Zhai et al."});
CREATE (mcro_siglipso400mpatch14384_UseCase:UseCaseInformationSection {id: "mcro:siglipso400mpatch14384-UseCase", hasTextValue: "zero-shot image classification and image-text retrieval"});
CREATE (mcro_siglipso400mpatch14384_Parameter:ModelParameterSection {id: "mcro:siglipso400mpatch14384-Parameter"});
CREATE (mcro_siglipso400mpatch14384_Dataset:DatasetInformationSection {id: "mcro:siglipso400mpatch14384-Dataset", hasTextValue: "WebLI dataset"});
CREATE (mcro_chronosboltsmall:Model {id: "mcro:chronosboltsmall"});
CREATE (mcro_chronosboltsmall_Citation:CitationInformationSection {id: "mcro:chronosboltsmall-Citation", hasTextValue: "@articleansari2024chronos,
    title=Chronos: Learning the Language of Time Series,
    author=Ansari, Abdul Fatir and Stella, Lorenzo and Turkmen, Caner and Zhang, Xiyuan, and Mercado, Pedro and Shen, Huibin and Shchur, Oleksandr and Rangapuram, Syama Syndar and Pineda Arango, Sebastian and Kapoor, Shubham and Zschiegner, Jasper and Maddix, Danielle C. and Mahoney, Michael W. and Torkkola, Kari and Gordon Wilson, Andrew and Bohlke-Schneider, Michael and Wang, Yuyang,
    journal=Transactions on Machine Learning Research,
    issn=2835-8856,
    year=2024,
    url=https://openreview.net/forum?id=gerNCVqqtR"});
CREATE (mcro_chronosboltsmall_License:LicenseInformationSection {id: "mcro:chronosboltsmall-License", hasTextValue: "This project is licensed under the Apache-2.0 License."});
CREATE (mcro_chronosboltsmall_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:chronosboltsmall-ModelArchitecture", hasTextValue: "It is based on the [T5 encoder-decoder architecture](https://arxiv.org/abs/1910.10683) and has been trained on nearly 100 billion time series observations."});
CREATE (mcro_chronosboltsmall_IntendedUseCase:UseCaseInformationSection {id: "mcro:chronosboltsmall-IntendedUseCase", hasTextValue: "pretrained time series forecasting models which can be used for zero-shot forecasting"});
CREATE (mcro_metaLlama31:Model {id: "mcro:metaLlama31"});
CREATE (mcro_metaLlama31_ModelDetail:ModelDetailSection {id: "mcro:metaLlama31-ModelDetail"});
CREATE (mcro_metaLlama31_License:LicenseInformationSection {id: "mcro:metaLlama31-License", hasTextValue: "A custom commercial license, the Llama 3.1 Community License"});
CREATE (mcro_metaLlama31_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:metaLlama31-ModelArchitecture", hasTextValue: "Llama 3.1 is an auto-regressive language model that uses an optimized transformer architecture."});
CREATE (mcro_metaLlama31_UseCase:UseCaseInformationSection {id: "mcro:metaLlama31-UseCase", hasTextValue: "Llama 3.1 is intended for commercial and research use in multiple languages."});
CREATE (mcro_metaLlama31_OutOfScopeUseCase:OutOfScopeUseCaseSectionInformation {id: "mcro:metaLlama31-OutOfScopeUseCase", hasTextValue: "Use in any manner that violates applicable laws or regulations (including trade compliance laws)."});
CREATE (mcro_metaLlama31_Dataset:DatasetInformationSection {id: "mcro:metaLlama31-Dataset", hasTextValue: "Llama 3.1 was pretrained on ~15 trillion tokens of data from publicly available sources."});
CREATE (mcro_distilbertbasemultilingualcased:Model {id: "mcro:distilbertbasemultilingualcased"});
CREATE (mcro_distilbertbasemultilingualcased_ConsiderationInformationSection:ConsiderationInformationSection {id: "mcro:distilbertbasemultilingualcased-ConsiderationInformationSection"});
CREATE (mcro_distilbertbasemultilingualcased_CitationInformationSection:CitationInformationSection {id: "mcro:distilbertbasemultilingualcased-CitationInformationSection"});
CREATE (mcro_distilbertbasemultilingualcased_DatasetInformationSection:DatasetInformationSection {id: "mcro:distilbertbasemultilingualcased-DatasetInformationSection"});
CREATE (mcro_distilbertbasemultilingualcased_LicenseInformationSection:LicenseInformationSection {id: "mcro:distilbertbasemultilingualcased-LicenseInformationSection", hasTextValue: "Apache 2.0"});
CREATE (mcro_distilbertbasemultilingualcased_LimitationInformationSection:LimitationInformationSection {id: "mcro:distilbertbasemultilingualcased-LimitationInformationSection"});
CREATE (mcro_distilbertbasemultilingualcased_ModelArchitectureInformationSection:ModelArchitectureInformationSection {id: "mcro:distilbertbasemultilingualcased-ModelArchitectureInformationSection"});
CREATE (mcro_distilbertbasemultilingualcased_QuantativeAnalysisSection:QuantativeAnalysisSection {id: "mcro:distilbertbasemultilingualcased-QuantativeAnalysisSection"});
CREATE (mcro_distilbertbasemultilingualcased_UseCaseInformationSection:UseCaseInformationSection {id: "mcro:distilbertbasemultilingualcased-UseCaseInformationSection"});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53portuguese:Model {id: "mcro:jonatasgrosmanwav2vec2largexlsr53portuguese"});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53portuguese_UseCase:UseCaseInformationSection {id: "mcro:jonatasgrosmanwav2vec2largexlsr53portuguese-UseCase", hasTextValue: "speech recognition in Portuguese"});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53portuguese_Architecture:ModelArchitectureInformationSection {id: "mcro:jonatasgrosmanwav2vec2largexlsr53portuguese-Architecture", hasTextValue: "XLSR-53 large model"});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53portuguese_Dataset:DatasetInformationSection {id: "mcro:jonatasgrosmanwav2vec2largexlsr53portuguese-Dataset", hasTextValue: "Common Voice 6.1"});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53portuguese_Citation:CitationInformationSection {id: "mcro:jonatasgrosmanwav2vec2largexlsr53portuguese-Citation", hasTextValue: "@miscgrosman2021xlsr53-large-portuguese,
  title=Fine-tuned XLSR-53 large model for speech recognition in Portuguese,
  author=Grosman, Jonatas,
  howpublished=https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-portuguese,
  year=2021"});
CREATE (mcro_XLMrobertalargeModel:Model {id: "mcro:XLMrobertalargeModel"});
CREATE (mcro_XLMrobertalargeModel_Citation:CitationInformationSection {id: "mcro:XLMrobertalargeModel-Citation", hasTextValue: "@articleDBLP:journals/corr/abs-1911-02116,
  author    = Alexis Conneau and
               Kartikay Khandelwal and
               Naman Goyal and
               Vishrav Chaudhary and
               Guillaume Wenzek and
               Francisco Guzm'an and
               Edouard Grave and
               Myle Ott and
               Luke Zettlemoyer and
               Veselin Stoyanov,
  title     = Unsupervised Cross-lingual Representation Learning at Scale,
  journal   = CoRR,
  volume    = abs/1911.02116,
  year      = 2019,
  url       = http://arxiv.org/abs/1911.02116,
  eprinttype = arXiv,
  eprint    = 1911.02116,
  timestamp = Mon, 11 Nov 2019 18:38:09 +0100,
  biburl    = https://dblp.org/rec/journals/corr/abs-1911-02116.bib,
  bibsource = dblp computer science bibliography, https://dblp.org"});
CREATE (mcro_XLMrobertalargeModel_Arch:ModelArchitectureInformationSection {id: "mcro:XLMrobertalargeModel-Arch", hasTextValue: "RoBERTa is a transformers model pretrained on a large corpus in a self-supervised fashion."});
CREATE (mcro_XLMrobertalargeModel_UseCase:UseCaseInformationSection {id: "mcro:XLMrobertalargeModel-UseCase", hasTextValue: "You can use the raw model for masked language modeling, but it's mostly intended to be fine-tuned on a downstream task."});
CREATE (mcro_clipvitlargepatch14336:Model {id: "mcro:clipvitlargepatch14336"});
CREATE (mcro_clipvitlargepatch14336_ModelArchitectureInformationSection:ModelArchitectureInformationSection {id: "mcro:clipvitlargepatch14336-ModelArchitectureInformationSection"});
CREATE (mcro_clipvitlargepatch14336_DatasetInformationSection:DatasetInformationSection {id: "mcro:clipvitlargepatch14336-DatasetInformationSection"});
CREATE (mcro_clipvitlargepatch14336_UseCaseInformationSection:UseCaseInformationSection {id: "mcro:clipvitlargepatch14336-UseCaseInformationSection"});
CREATE (mcro_esm2:Model {id: "mcro:esm2"});
CREATE (mcro_esm2_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:esm2-ModelArchitecture"});
CREATE (mcro_esm2_UseCase:UseCaseInformationSection {id: "mcro:esm2-UseCase"});
CREATE (mcro_esm2_Citation:CitationInformationSection {id: "mcro:esm2-Citation"});
CREATE (mcro_facebookesm2t4815BUR50D:Model {id: "mcro:facebookesm2t4815BUR50D"});
CREATE (mcro_facebookesm2t363BUR50D:Model {id: "mcro:facebookesm2t363BUR50D"});
CREATE (mcro_facebookesm2t33650MUR50D:Model {id: "mcro:facebookesm2t33650MUR50D"});
CREATE (mcro_facebookesm2t30150MUR50D:Model {id: "mcro:facebookesm2t30150MUR50D"});
CREATE (mcro_facebookesm2t1235MUR50D:Model {id: "mcro:facebookesm2t1235MUR50D"});
CREATE (mcro_facebookesm2t68MUR50D:Model {id: "mcro:facebookesm2t68MUR50D"});
CREATE (mcro_clip_PrimaryIntendedUseCaseInformationSection:PrimaryIntendedUseCaseInformationSection {id: "mcro:clip-PrimaryIntendedUseCaseInformationSection", hasTextValue: "The model is intended as a research output for research communities."});
CREATE (mcro_clip_OutofScopeUseCaseInformationSection:OutofScopeUseCaseInformationSection {id: "mcro:clip-OutofScopeUseCaseInformationSection", hasTextValue: "**Any** deployed use case of the model - whether commercial or not - is currently out of scope."});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53chinesezhcn:Model {id: "mcro:jonatasgrosmanwav2vec2largexlsr53chinesezhcn"});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53chinesezhcn_Citation:CitationInformationSection {id: "mcro:jonatasgrosmanwav2vec2largexlsr53chinesezhcn-Citation", hasTextValue: "@miscgrosman2021xlsr53-large-chinese,
  title=Fine-tuned XLSR-53 large model for speech recognition in Chinese,
  author=Grosman, Jonatas,
  howpublished=https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn,
  year=2021"});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53chinesezhcn_Arch:ModelArchitectureInformationSection {id: "mcro:jonatasgrosmanwav2vec2largexlsr53chinesezhcn-Arch", hasTextValue: "wav2vec2-large-xlsr-53"});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53chinesezhcn_Dataset:DatasetInformationSection {id: "mcro:jonatasgrosmanwav2vec2largexlsr53chinesezhcn-Dataset", hasTextValue: "ST-CMDS"});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53chinesezhcn_UseCase:UseCaseInformationSection {id: "mcro:jonatasgrosmanwav2vec2largexlsr53chinesezhcn-UseCase", hasTextValue: "speech recognition in Chinese"});
CREATE (mcro_t5base:Model {id: "mcro:t5base"});
CREATE (mcro_t5base_License:LicenseInformationSection {id: "mcro:t5base-License", hasTextValue: "Apache 2.0"});
CREATE (mcro_t5base_ModelDetails:ModelDetailSection {id: "mcro:t5base-ModelDetails"});
CREATE (mcro_t5base_UseCase:UseCaseInformationSection {id: "mcro:t5base-UseCase"});
CREATE (mcro_t5base_BiasRisksLimitations:ConsiderationInformationSection {id: "mcro:t5base-BiasRisksLimitations"});
CREATE (mcro_t5base_TrainingDetails:ModelParameterSection {id: "mcro:t5base-TrainingDetails"});
CREATE (mcro_t5base_Evaluation:QuantativeAnalysisSection {id: "mcro:t5base-Evaluation"});
CREATE (mcro_t5base_EnvironmentalImpact:ConsiderationInformationSection {id: "mcro:t5base-EnvironmentalImpact"});
CREATE (mcro_t5base_Citation:CitationInformationSection {id: "mcro:t5base-Citation"});
CREATE (mcro_t5base_ModelCardAuthors:OwnerInformationSection {id: "mcro:t5base-ModelCardAuthors"});
CREATE (mcro_distilbertbaseuncasedfinetunedsst2english:Model {id: "mcro:distilbertbaseuncasedfinetunedsst2english"});
CREATE (mcro_distilbertbaseuncasedfinetunedsst2english_ModelDetail:ModelDetailSection {id: "mcro:distilbertbaseuncasedfinetunedsst2english-ModelDetail", hasTextValue: "This model is a fine-tune checkpoint of [DistilBERT-base-uncased](https://huggingface.co/distilbert-base-uncased), fine-tuned on SST-2.
This model reaches an accuracy of 91.3 on the dev set (for comparison, Bert bert-base-uncased version reaches an accuracy of 92.7).
- **Developed by:** Hugging Face
- **Model Type:** Text Classification
- **Language(s):** English
- **License:** Apache-2.0
- **Parent Model:** For more details about DistilBERT, we encourage users to check out [this model card](https://huggingface.co/distilbert-base-uncased).
- **Resources for more information:**
    - [Model Documentation](https://huggingface.co/docs/transformers/main/en/model_doc/distilbert#transformers.DistilBertForSequenceClassification)
    - [DistilBERT paper](https://arxiv.org/abs/1910.01108)"});
CREATE (mcro_distilbertbaseuncasedfinetunedsst2english_UseCase:UseCaseInformationSection {id: "mcro:distilbertbaseuncasedfinetunedsst2english-UseCase", hasTextValue: "This model can be used for  topic classification. You can use the raw model for either masked language modeling or next sentence prediction, but it's mostly intended to be fine-tuned on a downstream task. See the model hub to look for fine-tuned versions on a task that interests you.

The model should not be used to intentionally create hostile or alienating environments for people. In addition, the model was not trained to be factual or true representations of people or events, and therefore using the model to generate such content is out-of-scope for the abilities of this model."});
CREATE (mcro_distilbertbaseuncasedfinetunedsst2english_Consideration:ConsiderationInformationSection {id: "mcro:distilbertbaseuncasedfinetunedsst2english-Consideration", hasTextValue: "Based on a few experimentations, we observed that this model could produce biased predictions that target underrepresented populations.

For instance, for sentences like `This film was filmed in COUNTRY`, this binary classification model will give radically different probabilities for the positive label depending on the country (0.89 if the country is France, but 0.08 if the country is Afghanistan) when nothing in the input indicates such a strong semantic shift. In this [colab](https://colab.research.google.com/gist/ageron/fb2f64fb145b4bc7c49efc97e5f114d3/biasmap.ipynb), [Aurélien Géron](https://twitter.com/aureliengeron) made an interesting map plotting these probabilities for each country.


We strongly advise users to thoroughly probe these aspects on their use-cases in order to evaluate the risks of this model. We recommend looking at the following bias evaluation datasets as a place to start: [WinoBias](https://huggingface.co/datasets/wino_bias), [WinoGender](https://huggingface.co/datasets/super_glue), [Stereoset](https://huggingface.co/datasets/stereoset)."});
CREATE (mcro_distilbertbaseuncasedfinetunedsst2english_Training:ModelParameterSection {id: "mcro:distilbertbaseuncasedfinetunedsst2english-Training", hasTextValue: "The authors use the following Stanford Sentiment Treebank([sst2](https://huggingface.co/datasets/sst2)) corpora for the model.

- learning_rate = 1e-5
- batch_size = 32
- warmup = 600
- max_seq_length = 128
- num_train_epochs = 3.0"});
CREATE (mcro_distilbertbaseuncasedfinetunedsst2english_License:LicenseInformationSection {id: "mcro:distilbertbaseuncasedfinetunedsst2english-License", hasTextValue: "Apache-2.0"});
CREATE (mcro_multilinguale5small:Model {id: "mcro:multilinguale5small"});
CREATE (mcro_multilinguale5small_CitationInformationSection:CitationInformationSection {id: "mcro:multilinguale5small-CitationInformationSection", hasTextValue: "@articlewang2024multilingual,
  title=Multilingual E5 Text Embeddings: A Technical Report,
  author=Wang, Liang and Yang, Nan and Huang, Xiaolong and Yang, Linjun and Majumder, Rangan and Wei, Furu,
  journal=arXiv preprint arXiv:2402.05672,
  year=2024"});
CREATE (mcro_multilinguale5small_ModelArchitectureInformationSection:ModelArchitectureInformationSection {id: "mcro:multilinguale5small-ModelArchitectureInformationSection", hasTextValue: "This model has 12 layers and the embedding size is 384."});
CREATE (mcro_multilinguale5small_TrainingDataInformationSection:TrainingDataInformationSection {id: "mcro:multilinguale5small-TrainingDataInformationSection", hasTextValue: "This model is initialized from [microsoft/Multilingual-MiniLM-L12-H384](https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384)
and continually trained on a mixture of multilingual datasets."});
CREATE (mcro_multilinguale5small_BenchmarkResults:IAO_0000310 {id: "mcro:multilinguale5small-BenchmarkResults"});
CREATE (mcro_multilinguale5small_LimitationInformationSection:LimitationInformationSection {id: "mcro:multilinguale5small-LimitationInformationSection", hasTextValue: "Long texts will be truncated to at most 512 tokens."});
CREATE (mcro_multilinguale5small_IntendedUseCaseInformationSection:UseCaseInformationSection {id: "mcro:multilinguale5small-IntendedUseCaseInformationSection"});
CREATE (mcro_visiontransformerbase:Model {id: "mcro:visiontransformerbase"});
CREATE (mcro_visiontransformerbase_ModelDetail:ModelDetailSection {id: "mcro:visiontransformerbase-ModelDetail"});
CREATE (mcro_visiontransformerbase_Citation:CitationInformationSection {id: "mcro:visiontransformerbase-Citation"});
CREATE (mcro_visiontransformerbase_License:LicenseInformationSection {id: "mcro:visiontransformerbase-License"});
CREATE (mcro_visiontransformerbase_ModelParameter:ModelParameterSection {id: "mcro:visiontransformerbase-ModelParameter"});
CREATE (mcro_visiontransformerbase_Dataset:DatasetInformationSection {id: "mcro:visiontransformerbase-Dataset"});
CREATE (mcro_visiontransformerbase_IntendedUseCase:UseCaseInformationSection {id: "mcro:visiontransformerbase-IntendedUseCase", hasTextValue: "image classification"});
CREATE (mcro_visiontransformerbase_TrainingData:DatasetInformationSection:TrainingDataInformationSection {id: "mcro:visiontransformerbase-TrainingData", hasTextValue: "ImageNet-21k"});
CREATE (mcro_visiontransformerbase_TrainingProcedure:ModelParameterSection {id: "mcro:visiontransformerbase-TrainingProcedure"});
CREATE (mcro_visiontransformerbase_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:visiontransformerbase-ModelArchitecture", hasTextValue: "Vision Transformer (ViT) model pre-trained on ImageNet-21k (14 million images, 21,843 classes) at resolution 224x224"});
CREATE (mcro_bertbasemodelcased:Model {id: "mcro:bertbasemodelcased"});
CREATE (mcro_bertbasemodelcased_ModelDetail:ModelDetailSection {id: "mcro:bertbasemodelcased-ModelDetail"});
CREATE (mcro_bertbasemodelcased_Citation:CitationInformationSection {id: "mcro:bertbasemodelcased-Citation"});
CREATE (mcro_bertbasemodelcased_Architecture:ModelArchitectureInformationSection {id: "mcro:bertbasemodelcased-Architecture", hasTextValue: "transformers"});
CREATE (mcro_bertbasemodelcased_License:LicenseInformationSection {id: "mcro:bertbasemodelcased-License"});
CREATE (mcro_bertbasemodelcased_UseCase:UseCaseInformationSection {id: "mcro:bertbasemodelcased-UseCase"});
CREATE (mcro_bertbasemodelcased_Consideration:ConsiderationInformationSection {id: "mcro:bertbasemodelcased-Consideration"});
CREATE (mcro_bertbasemodelcased_TrainingData:TrainingDataInformationSection {id: "mcro:bertbasemodelcased-TrainingData", hasTextValue: "English Wikipedia"});
CREATE (mcro_bertbasemodelcased_Parameter:ModelParameterSection {id: "mcro:bertbasemodelcased-Parameter", hasTextValue: "Adam"});
CREATE (mcro_bertbasemodelcased_Evaluation:QuantativeAnalysisSection {id: "mcro:bertbasemodelcased-Evaluation"});
CREATE (mcro_bertbasemodelcased_Dataset:EvaluationDataInformationSection {id: "mcro:bertbasemodelcased-Dataset"});
CREATE (mcro_jinaaijinaembeddingsv3:Model {id: "mcro:jinaaijinaembeddingsv3"});
CREATE (mcro_jinaaijinaembeddingsv3_UseCaseInformationSection:UseCaseInformationSection {id: "mcro:jinaaijinaembeddingsv3-UseCaseInformationSection", hasTextValue: "`jina-embeddings-v3` is a **multilingual multi-task text embedding model** designed for a variety of NLP applications."});
CREATE (mcro_jinaaijinaembeddingsv3_ModelArchitectureInformationSection:ModelArchitectureInformationSection {id: "mcro:jinaaijinaembeddingsv3-ModelArchitectureInformationSection", hasTextValue: "Based on the [Jina-XLM-RoBERTa architecture](https://huggingface.co/jinaai/xlm-roberta-flash-implementation), 
this model supports Rotary Position Embeddings to handle long input sequences up to **8192 tokens**."});
CREATE (mcro_jinaaijinaembeddingsv3_LicenseInformationSection:LicenseInformationSection {id: "mcro:jinaaijinaembeddingsv3-LicenseInformationSection", hasTextValue: "CC BY-NC 4.0"});
CREATE (mcro_jinaaijinaembeddingsv3_CitationInformationSection:CitationInformationSection {id: "mcro:jinaaijinaembeddingsv3-CitationInformationSection", hasTextValue: "@miscsturua2024jinaembeddingsv3multilingualembeddingstask,
      title=jina-embeddings-v3: Multilingual Embeddings With Task LoRA, 
      author=Saba Sturua and Isabelle Mohr and Mohammad Kalim Akram and Michael Günther and Bo Wang and Markus Krimmel and Feng Wang and Georgios Mastrapas and Andreas Koukounas and Andreas Koukounas and Nan Wang and Han Xiao,
      year=2024,
      eprint=2409.10173,
      archivePrefix=arXiv,
      primaryClass=cs.CL,
      url=https://arxiv.org/abs/2409.10173,"});
CREATE (mcro_sentencetransformersparaphraseMiniLML6v2:Model {id: "mcro:sentencetransformersparaphraseMiniLML6v2"});
CREATE (mcro_sentencetransformersparaphraseMiniLML6v2_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:sentencetransformersparaphraseMiniLML6v2-ModelArchitecture", hasTextValue: "Transformer('max_seq_length': 128, 'do_lower_case': False) with Transformer model: BertModel"});
CREATE (mcro_sentencetransformersparaphraseMiniLML6v2_Citation:CitationInformationSection {id: "mcro:sentencetransformersparaphraseMiniLML6v2-Citation", hasTextValue: "@inproceedingsreimers-2019-sentence-bert,
    title = \"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks\",
    author = \"Reimers, Nils and Gurevych, Iryna\",
    booktitle = \"Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing\",
    month = \"11\",
    year = \"2019\",
    publisher = \"Association for Computational Linguistics\",
    url = \"http://arxiv.org/abs/1908.10084\","});
CREATE (mcro_resnet18a1in1k:Model {id: "mcro:resnet18a1in1k"});
CREATE (mcro_resnet18a1in1k_ModelDetail:ModelDetailSection {id: "mcro:resnet18a1in1k-ModelDetail"});
CREATE (mcro_resnet18a1in1k_Citation:CitationInformationSection {id: "mcro:resnet18a1in1k-Citation", hasTextValue: "@articleHe2015,
  author = Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun,
  title = Deep Residual Learning for Image Recognition,
  journal = arXiv preprint arXiv:1512.03385,
  year = 2015"});
CREATE (mcro_resnet18a1in1k_Architecture:ModelArchitectureInformationSection {id: "mcro:resnet18a1in1k-Architecture", hasTextValue: "1x1 convolution shortcut downsample"});
CREATE (mcro_resnet18a1in1k_Dataset:DatasetInformationSection {id: "mcro:resnet18a1in1k-Dataset", hasTextValue: "ImageNet-1k"});
CREATE (mcro_resnet18a1in1k_UseCase:UseCaseInformationSection {id: "mcro:resnet18a1in1k-UseCase", hasTextValue: "Image classification / feature backbone"});
CREATE (mcro_flant5base:Model {id: "mcro:flant5base"});
CREATE (mcro_flant5base_ModelDetail:ModelDetailSection {id: "mcro:flant5base-ModelDetail"});
CREATE (mcro_flant5base_License:LicenseInformationSection {id: "mcro:flant5base-License", hasTextValue: "Apache 2.0"});
CREATE (mcro_flant5base_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:flant5base-ModelArchitecture", hasTextValue: "Language model"});
CREATE (mcro_flant5base_Citation:CitationInformationSection {id: "mcro:flant5base-Citation", hasTextValue: "@mischttps://doi.org/10.48550/arxiv.2210.11416,
  doi = 10.48550/ARXIV.2210.11416,
  
  url = https://arxiv.org/abs/2210.11416,
  
  author = Chung, Hyung Won and Hou, Le and Longpre, Shayne and Zoph, Barret and Tay, Yi and Fedus, William and Li, Eric and Wang, Xuezhi and Dehghani, Mostafa and Brahma, Siddhartha and Webson, Albert and Gu, Shixiang Shane and Dai, Zhuyun and Suzgun, Mirac and Chen, Xinyun and Chowdhery, Aakanksha and Narang, Sharan and Mishra, Gaurav and Yu, Adams and Zhao, Vincent and Huang, Yanping and Dai, Andrew and Yu, Hongkun and Petrov, Slav and Chi, Ed H. and Dean, Jeff and Devlin, Jacob and Roberts, Adam and Zhou, Denny and Le, Quoc V. and Wei, Jason,
  
  keywords = Machine Learning (cs.LG), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences,
  
  title = Scaling Instruction-Finetuned Language Models,
  
  publisher = arXiv,
  
  year = 2022,
  
  copyright = Creative Commons Attribution 4.0 International"});
CREATE (mcro_flant5base_UseCase:UseCaseInformationSection {id: "mcro:flant5base-UseCase", hasTextValue: "The primary use is research on language models, including: research on zero-shot NLP tasks and in-context few-shot learning NLP tasks, such as reasoning, and question answering; advancing fairness and safety research, and understanding limitations of current large language models"});
CREATE (mcro_fashionclip:Model {id: "mcro:fashionclip"});
CREATE (mcro_fashionclip_ModelDetail:ModelDetailSection {id: "mcro:fashionclip-ModelDetail"});
CREATE (mcro_fashionclip_Citation:CitationInformationSection {id: "mcro:fashionclip-Citation", hasTextValue: "@ArticleChia2022,
    title=\"Contrastive language and vision learning of general fashion concepts\",
    author=\"Chia, Patrick John
            and Attanasio, Giuseppe
            and Bianchi, Federico
            and Terragni, Silvia
            and Magalh~aes, Ana Rita
            and Goncalves, Diogo
            and Greco, Ciro
            and Tagliabue, Jacopo\",
    journal=\"Scientific Reports\",
    year=\"2022\",
    month=\"Nov\",
    day=\"08\",
    volume=\"12\",
    number=\"1\",
    abstract=\"The steady rise of online shopping goes hand in hand with the development of increasingly complex ML and NLP models. While most use cases are cast as specialized supervised learning problems, we argue that practitioners would greatly benefit from general and transferable representations of products. In this work, we build on recent developments in contrastive learning to train FashionCLIP, a CLIP-like model adapted for the fashion industry. We demonstrate the effectiveness of the representations learned by FashionCLIP with extensive tests across a variety of tasks, datasets and generalization probes. We argue that adaptations of large pre-trained models such as CLIP offer new perspectives in terms of scalability and sustainability for certain types of players in the industry. Finally, we detail the costs and environmental impact of training, and release the model weights and code as open source contribution to the community.\",
    issn=\"2045-2322\",
    doi=\"10.1038/s41598-022-23052-9\",
    url=\"https://doi.org/10.1038/s41598-022-23052-9\""});
CREATE (mcro_fashionclip_Architecture:ModelArchitectureInformationSection {id: "mcro:fashionclip-Architecture", hasTextValue: "The model uses a ViT-B/32 Transformer architecture as an image encoder and uses a masked self-attention Transformer as a text encoder. These encoders are trained, starting from a pre-trained checkpoint, to maximize the similarity of (image, text) pairs via a contrastive loss on a fashion dataset containing 800K products."});
CREATE (mcro_fashionclip_Dataset:DatasetInformationSection {id: "mcro:fashionclip-Dataset", hasTextValue: "The model was trained on (image, text) pairs obtained from the Farfecth dataset[^1 Awaiting official release.], an English dataset comprising over 800K fashion products, with more than 3K brands across dozens of object types. The image used for encoding is the standard product image, which is a picture of the item over a white background, with no humans. The text used is a concatenation of the _highlight_ (e.g., “stripes”, “long sleeves”, “Armani”) and _short description_ (“80s styled t-shirt”)) available in the Farfetch dataset."});
CREATE (mcro_fashionclip_Consideration:ConsiderationInformationSection {id: "mcro:fashionclip-Consideration", hasTextValue: "We acknowledge certain limitations of FashionCLIP and expect that it inherits certain limitations and biases present in the original CLIP model. We do not expect our fine-tuning to significantly augment these limitations: we acknowledge that the fashion data we use makes explicit assumptions about the notion of gender as in \"blue shoes for a woman\" that inevitably associate aspects of clothing with specific people.

Our investigations also suggest that the data used introduces certain limitations in FashionCLIP. From the textual modality, given that most captions derived from the Farfetch dataset are long, we observe that FashionCLIP may be more performant in longer queries than shorter ones. From the image modality, FashionCLIP is also biased towards standard product images (centered, white background).

Model selection, i.e. selecting an appropariate stopping critera during fine-tuning, remains an open challenge. We observed that using loss on an in-domain (i.e. same distribution as test) validation dataset is a poor selection critera when out-of-domain generalization (i.e. across different datasets) is desired, even when the dataset used is relatively diverse and large."});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53russian:Model {id: "mcro:jonatasgrosmanwav2vec2largexlsr53russian"});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53russian_Citation:CitationInformationSection {id: "mcro:jonatasgrosmanwav2vec2largexlsr53russian-Citation", hasTextValue: "@miscgrosman2021xlsr53-large-russian,
  title=Fine-tuned XLSR-53 large model for speech recognition in Russian,
  author=Grosman, Jonatas,
  howpublished=https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-russian,
  year=2021"});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53russian_Architecture:ModelArchitectureInformationSection {id: "mcro:jonatasgrosmanwav2vec2largexlsr53russian-Architecture", hasTextValue: "Wav2Vec2ForCTC"});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53russian_Dataset:DatasetInformationSection {id: "mcro:jonatasgrosmanwav2vec2largexlsr53russian-Dataset", hasTextValue: "CSS10"});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53russian_UseCase:UseCaseInformationSection {id: "mcro:jonatasgrosmanwav2vec2largexlsr53russian-UseCase", hasTextValue: "speech recognition in Russian"});
CREATE (mcro_twitterrobertabasesentiment:Model {id: "mcro:twitterrobertabasesentiment"});
CREATE (mcro_twitterrobertabasesentiment_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:twitterrobertabasesentiment-ModelArchitecture", hasTextValue: "roBERTa-base"});
CREATE (mcro_twitterrobertabasesentiment_Dataset:DatasetInformationSection {id: "mcro:twitterrobertabasesentiment-Dataset", hasTextValue: "~58M tweets"});
CREATE (mcro_twitterrobertabasesentiment_Citation:CitationInformationSection {id: "mcro:twitterrobertabasesentiment-Citation", hasTextValue: "_TweetEval_ (Findings of EMNLP 2020)"});
CREATE (mcro_twitterrobertabasesentiment_IntendedUseCase:UseCaseInformationSection {id: "mcro:twitterrobertabasesentiment-IntendedUseCase", hasTextValue: "sentiment analysis"});
CREATE (mcro_vitmatte:Model {id: "mcro:vitmatte"});
CREATE (mcro_vitmatte_ModelDetail:ModelDetailSection {id: "mcro:vitmatte-ModelDetail"});
CREATE (mcro_vitmatte_Citation:CitationInformationSection {id: "mcro:vitmatte-Citation", hasTextValue: "@miscyao2023vitmatte,
      title=ViTMatte: Boosting Image Matting with Pretrained Plain Vision Transformers, 
      author=Jingfeng Yao and Xinggang Wang and Shusheng Yang and Baoyuan Wang,
      year=2023,
      eprint=2305.15272,
      archivePrefix=arXiv,
      primaryClass=cs.CV"});
CREATE (mcro_vitmatte_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:vitmatte-ModelArchitecture", hasTextValue: "Vision Transformer (ViT) with a lightweight head on top"});
CREATE (mcro_vitmatte_UseCase:UseCaseInformationSection {id: "mcro:vitmatte-UseCase", hasTextValue: "image matting"});
CREATE (mcro_FlagEmbedding:Model {id: "mcro:FlagEmbedding"});
CREATE (mcro_FlagEmbedding_Citation:CitationInformationSection {id: "mcro:FlagEmbedding-Citation", hasTextValue: "@miscbge_embedding,
      title=C-Pack: Packaged Resources To Advance General Chinese Embedding, 
      author=Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff,
      year=2023,
      eprint=2309.07597,
      archivePrefix=arXiv,
      primaryClass=cs.CL"});
CREATE (mcro_FlagEmbedding_License:LicenseInformationSection {id: "mcro:FlagEmbedding-License", hasTextValue: "MIT License"});
CREATE (mcro_bartlargecnn:Model {id: "mcro:bartlargecnn"});
CREATE (mcro_bartlargecnn_ModelDetail:ModelDetailSection {id: "mcro:bartlargecnn-ModelDetail"});
CREATE (mcro_bartlargecnn_Citation:CitationInformationSection {id: "mcro:bartlargecnn-Citation", hasTextValue: "@articleDBLP:journals/corr/abs-1910-13461,
  author    = Mike Lewis and
               Yinhan Liu and
               Naman Goyal and
               Marjan Ghazvininejad and
               Abdelrahman Mohamed and
               Omer Levy and
               Veselin Stoyanov and
               Luke Zettlemoyer,
  title     = BART: Denoising Sequence-to-Sequence Pre-training for Natural Language
               Generation, Translation, and Comprehension,
  journal   = CoRR,
  volume    = abs/1910.13461,
  year      = 2019,
  url       = http://arxiv.org/abs/1910.13461,
  eprinttype = arXiv,
  eprint    = 1910.13461,
  timestamp = Thu, 31 Oct 2019 14:02:26 +0100,
  biburl    = https://dblp.org/rec/journals/corr/abs-1910-13461.bib,
  bibsource = dblp computer science bibliography, https://dblp.org"});
CREATE (mcro_bartlargecnn_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:bartlargecnn-ModelArchitecture", hasTextValue: "BART is a transformer encoder-encoder (seq2seq) model with a bidirectional (BERT-like) encoder and an autoregressive (GPT-like) decoder."});
CREATE (mcro_bartlargecnn_UseCase:UseCaseInformationSection {id: "mcro:bartlargecnn-UseCase", hasTextValue: "You can use this model for text summarization."});
CREATE (mcro_bartlargecnn_Dataset:DatasetInformationSection {id: "mcro:bartlargecnn-Dataset", hasTextValue: "CNN Daily Mail"});
CREATE (mcro_stablediffusionv15modelcard:Model {id: "mcro:stablediffusionv15modelcard"});
CREATE (mcro_stablediffusionv15modelcard_License:LicenseInformationSection {id: "mcro:stablediffusionv15modelcard-License", hasTextValue: "The CreativeML OpenRAIL M license is an Open RAIL M license, adapted from the work that BigScience and the RAIL Initiative are jointly carrying in the area of responsible AI licensing."});
CREATE (mcro_stablediffusionv15modelcard_Citation:CitationInformationSection {id: "mcro:stablediffusionv15modelcard-Citation", hasTextValue: "@InProceedingsRombach_2022_CVPR,
          author    = Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn,
          title     = High-Resolution Image Synthesis With Latent Diffusion Models,
          booktitle = Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
          month     = June,
          year      = 2022,
          pages     = 10684-10695"});
CREATE (mcro_stablediffusionv15modelcard_IntendedUseCase:UseCaseInformationSection {id: "mcro:stablediffusionv15modelcard-IntendedUseCase", hasTextValue: "The model is intended for research purposes only. Possible research areas and
tasks include

- Safe deployment of models which have the potential to generate harmful content.
- Probing and understanding the limitations and biases of generative models.
- Generation of artworks and use in design and other artistic processes.
- Applications in educational or creative tools.
- Research on generative models."});
CREATE (mcro_stablediffusionv15modelcard_Architecture:ModelArchitectureInformationSection {id: "mcro:stablediffusionv15modelcard-Architecture", hasTextValue: "Diffusion-based text-to-image generation model"});
CREATE (mcro_stablediffusionv15modelcard_Dataset:DatasetInformationSection {id: "mcro:stablediffusionv15modelcard-Dataset", hasTextValue: "LAION-2B (en) and subsets thereof"});
CREATE (mcro_BGE_M3:Model {id: "mcro:BGE-M3"});
CREATE (mcro_BGE_M3_Citation:CitationInformationSection {id: "mcro:BGE-M3-Citation", hasTextValue: "@miscbge-m3,
      title=BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation, 
      author=Jianlv Chen and Shitao Xiao and Peitian Zhang and Kun Luo and Defu Lian and Zheng Liu,
      year=2024,
      eprint=2402.03216,
      archivePrefix=arXiv,
      primaryClass=cs.CL"});
CREATE (mcro_BGE_M3_Architecture:ModelArchitectureInformationSection {id: "mcro:BGE-M3-Architecture", hasTextValue: "BGE-M3"});
CREATE (mcro_BGE_M3_UseCase:UseCaseInformationSection {id: "mcro:BGE-M3-UseCase", hasTextValue: "dense retrieval, multi-vector retrieval, and sparse retrieval"});
CREATE (mcro_BGE_M3_Dataset:DatasetInformationSection {id: "mcro:BGE-M3-Dataset", hasTextValue: "bge-m3-data"});
CREATE (mcro_YOLOWorldMirror:Model {id: "mcro:YOLOWorldMirror"});
CREATE (mcro_YOLOWorldMirror_Documentation:DocumentationSection {id: "mcro:YOLOWorldMirror-Documentation", hasTextValue: "https://docs.ultralytics.com/models/yolo-world/#available-models-supported-tasks-and-operating-modes

model weights for ultralytics yolo models"});
CREATE (mcro_bertbasechinese:Model {id: "mcro:bertbasechinese"});
CREATE (mcro_bertbasechinese_ModelDetailSection:ModelDetailSection {id: "mcro:bertbasechinese-ModelDetailSection"});
CREATE (mcro_bertbasechinese_License:LicenseInformationSection {id: "mcro:bertbasechinese-License"});
CREATE (mcro_bertbasechinese_UseCaseSection:UseCaseInformationSection {id: "mcro:bertbasechinese-UseCaseSection"});
CREATE (mcro_bertbasechinese_ConsiderationSection:ConsiderationInformationSection {id: "mcro:bertbasechinese-ConsiderationSection"});
CREATE (mcro_bertbasechinese_ModelParameterSection:ModelParameterSection {id: "mcro:bertbasechinese-ModelParameterSection"});
CREATE (mcro_bertbasechinese_QuantativeAnalysisSection:QuantativeAnalysisSection {id: "mcro:bertbasechinese-QuantativeAnalysisSection"});
CREATE (mcro_bartlargemnli:Model {id: "mcro:bartlargemnli"});
CREATE (mcro_bartlargemnli_Citation:CitationInformationSection {id: "mcro:bartlargemnli-Citation", hasTextValue: "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"});
CREATE (mcro_facebookbartlargemnli_Citation2:CitationInformationSection {id: "mcro:facebookbartlargemnli-Citation2", hasTextValue: "BART fairseq implementation"});
CREATE (mcro_MultiNLI:DatasetInformationSection {id: "mcro:MultiNLI", hasTextValue: "MultiNLI (MNLI)"});
CREATE (mcro_BART:ModelArchitectureInformationSection {id: "mcro:BART", hasTextValue: "bart-large"});
CREATE (mcro_NLIbasedZeroShotTextClassification:UseCaseInformationSection {id: "mcro:NLIbasedZeroShotTextClassification", hasTextValue: "NLI-based Zero Shot Text Classification"});
CREATE (mcro_CLIPViTB16LAION2B:Model {id: "mcro:CLIPViTB16LAION2B"});
CREATE (mcro_CLIPViTB16LAION2B_DatasetInfo:DatasetInformationSection {id: "mcro:CLIPViTB16LAION2B-DatasetInfo", hasTextValue: "This model was trained with the 2 Billion sample English subset of LAION-5B (https://laion.ai/blog/laion-5b/)."});
CREATE (mcro_CLIPViTB16LAION2B_ModelArch:ModelArchitectureInformationSection {id: "mcro:CLIPViTB16LAION2B-ModelArch", hasTextValue: "CLIP ViT-B/16"});
CREATE (mcro_CLIPViTB16LAION2B_LicenseInfo:LicenseInformationSection {id: "mcro:CLIPViTB16LAION2B-LicenseInfo"});
CREATE (mcro_CLIPViTB16LAION2B_CitationInfo:CitationInformationSection {id: "mcro:CLIPViTB16LAION2B-CitationInfo"});
CREATE (mcro_CLIPViTB16LAION2B_UseCaseInfo:UseCaseInformationSection {id: "mcro:CLIPViTB16LAION2B-UseCaseInfo", hasTextValue: "research output for research communities. We hope that this model will enable researchers to better understand and explore zero-shot, arbitrary image classification. We also hope it can be used for interdisciplinary studies of the potential impact of such model."});
CREATE (mcro_stablediffusioninpaintingmodelcard:Model:ModelCardReport {id: "mcro:stablediffusioninpaintingmodelcard"});
CREATE (mcro_stablediffusioninpaintingmodelcard_ModelDetail:ModelDetailSection {id: "mcro:stablediffusioninpaintingmodelcard-ModelDetail"});
CREATE (mcro_stablediffusioninpaintingmodelcard_UseCase:UseCaseInformationSection {id: "mcro:stablediffusioninpaintingmodelcard-UseCase"});
CREATE (mcro_stablediffusioninpaintingmodelcard_Limitation:LimitationInformationSection {id: "mcro:stablediffusioninpaintingmodelcard-Limitation"});
CREATE (mcro_stablediffusioninpaintingmodelcard_TrainingData:TrainingDataInformationSection {id: "mcro:stablediffusioninpaintingmodelcard-TrainingData"});
CREATE (mcro_stablediffusioninpaintingmodelcard_EvaluationResult:QuantativeAnalysisSection {id: "mcro:stablediffusioninpaintingmodelcard-EvaluationResult"});
CREATE (mcro_stablediffusioninpaintingmodelcard_Citation:CitationInformationSection {id: "mcro:stablediffusioninpaintingmodelcard-Citation", hasTextValue: "@InProceedingsRombach_2022_CVPR,
 author    = Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn,
 title     = High-Resolution Image Synthesis With Latent Diffusion Models,
 booktitle = Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR),
 month     = June,
 year      = 2022,
 pages     = 10684-10695"});
CREATE (mcro_stablediffusioninpaintingmodelcard_License:LicenseInformationSection {id: "mcro:stablediffusioninpaintingmodelcard-License", hasTextValue: "The CreativeML OpenRAIL M license"});
CREATE (mcro_stablediffusioninpaintingmodelcard_Architecture:ModelArchitectureInformationSection {id: "mcro:stablediffusioninpaintingmodelcard-Architecture", hasTextValue: "Diffusion-based text-to-image generation model"});
CREATE (mcro_stablediffusioninpaintingmodelcard_Dataset:DatasetInformationSection {id: "mcro:stablediffusioninpaintingmodelcard-Dataset", hasTextValue: "LAION-2B (en)"});
CREATE (mcro_Qwen2505BInstruct:Model {id: "mcro:Qwen2505BInstruct"});
CREATE (mcro_Qwen2505BInstruct_UseCase:UseCaseInformationSection {id: "mcro:Qwen2505BInstruct-UseCase", hasTextValue: "This model is intended for use in the [Gensyn RL Swarm](https://www.gensyn.ai/articles/rl-swarm), to finetune locally using peer-to-peer reinforcement learning post-training.

Once finetuned, the model can be used as normal in any workflow, for details on how to do this please refer to the [original model documentation](https://qwen.readthedocs.io/en/latest/).

For more details on the original model, please refer to the original repository [here](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)."});
CREATE (mcro_Qwen2505BInstruct_Architecture:ModelArchitectureInformationSection {id: "mcro:Qwen2505BInstruct-Architecture", hasTextValue: "transformers with RoPE, SwiGLU, RMSNorm, Attention QKV bias and tied word embeddings"});
CREATE (mcro_Qwen2505BInstruct_Parameter:ModelParameterSection {id: "mcro:Qwen2505BInstruct-Parameter", hasTextValue: "- Number of Parameters: 0.49B
- Number of Paramaters (Non-Embedding): 0.36B
- Number of Layers: 24
- Number of Attention Heads (GQA): 14 for Q and 2 for KV
- Context Length: Full 32,768 tokens and generation 8192 tokens"});
CREATE (mcro_Qwen257BInstruct:Model {id: "mcro:Qwen257BInstruct"});
CREATE (mcro_Qwen257BInstruct_Architecture:ModelArchitectureInformationSection {id: "mcro:Qwen257BInstruct-Architecture", hasTextValue: "transformers with RoPE, SwiGLU, RMSNorm, and Attention QKV bias"});
CREATE (mcro_Qwen257BInstruct_Citation:CitationInformationSection {id: "mcro:Qwen257BInstruct-Citation", hasTextValue: "@miscqwen2.5,
    title = Qwen2.5: A Party of Foundation Models,
    url = https://qwenlm.github.io/blog/qwen2.5/,
    author = Qwen Team,
    month = September,
    year = 2024


@articleqwen2,
      title=Qwen2 Technical Report, 
      author=An Yang and Baosong Yang and Binyuan Hui and Bo Zheng and Bowen Yu and Chang Zhou and Chengpeng Li and Chengyuan Li and Dayiheng Liu and Fei Huang and Guanting Dong and Haoran Wei and Huan Lin and Jialong Tang and Jialin Wang and Jian Yang and Jianhong Tu and Jianwei Zhang and Jianxin Ma and Jin Xu and Jingren Zhou and Jinze Bai and Jinzheng He and Junyang Lin and Kai Dang and Keming Lu and Keqin Chen and Kexin Yang and Mei Li and Mingfeng Xue and Na Ni and Pei Zhang and Peng Wang and Ru Peng and Rui Men and Ruize Gao and Runji Lin and Shijie Wang and Shuai Bai and Sinan Tan and Tianhang Zhu and Tianhao Li and Tianyu Liu and Wenbin Ge and Xiaodong Deng and Xiaohuan Zhou and Xingzhang Ren and Xinyu Zhang and Xipin Wei and Xuancheng Ren and Yang Fan and Yang Yao and Yichang Zhang and Yu Wan and Yunfei Chu and Yuqiong Liu and Zeyu Cui and Zhenru Zhang and Zhihao Fan,
      journal=arXiv preprint arXiv:2407.10671,
      year=2024"});
CREATE (mcro_clipViTbigG14laion2B:Model {id: "mcro:clipViTbigG14laion2B"});
CREATE (mcro_clipViTbigG14laion2B_ModelDetailSection:ModelDetailSection {id: "mcro:clipViTbigG14laion2B-ModelDetailSection"});
CREATE (mcro_clipViTbigG14laion2B_License:LicenseInformationSection {id: "mcro:clipViTbigG14laion2B-License", hasTextValue: "MIT"});
CREATE (mcro_clipViTbigG14laion2B_CitationLAION:CitationInformationSection {id: "mcro:clipViTbigG14laion2B-CitationLAION", hasTextValue: "@inproceedingsschuhmann2022laionb,
  title=LAION-5B: An open large-scale dataset for training next generation image-text models,
  author=Christoph Schuhmann and
          Romain Beaumont and
          Richard Vencu and
          Cade W Gordon and
          Ross Wightman and
          Mehdi Cherti and
          Theo Coombes and
          Aarush Katta and
          Clayton Mullis and
          Mitchell Wortsman and
          Patrick Schramowski and
          Srivatsa R Kundurthy and
          Katherine Crowson and
          Ludwig Schmidt and
          Robert Kaczmarczyk and
          Jenia Jitsev,
  booktitle=Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track,
  year=2022,
  url=https://openreview.net/forum?id=M3Y74vmsMcY"});
CREATE (mcro_clipViTbigG14laion2B_CitationOpenAICLIP:CitationInformationSection {id: "mcro:clipViTbigG14laion2B-CitationOpenAICLIP", hasTextValue: "@inproceedingsRadford2021LearningTV,
  title=Learning Transferable Visual Models From Natural Language Supervision,
  author=Alec Radford and Jong Wook Kim and Chris Hallacy and A. Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever,
  booktitle=ICML,
  year=2021"});
CREATE (mcro_clipViTbigG14laion2B_CitationOpenCLIP:CitationInformationSection {id: "mcro:clipViTbigG14laion2B-CitationOpenCLIP", hasTextValue: "@softwareilharco_gabriel_2021_5143773,
  author       = Ilharco, Gabriel and
                  Wortsman, Mitchell and
                  Wightman, Ross and
                  Gordon, Cade and
                  Carlini, Nicholas and
                  Taori, Rohan and
                  Dave, Achal and
                  Shankar, Vaishaal and
                  Namkoong, Hongseok and
                  Miller, John and
                  Hajishirzi, Hannaneh and
                  Farhadi, Ali and
                  Schmidt, Ludwig,
  title        = OpenCLIP,
  month        = jul,
  year         = 2021,
  note         = If you use this software, please cite it as below.,
  publisher    = Zenodo,
  version      = 0.1,
  doi          = 10.5281/zenodo.5143773,
  url          = https://doi.org/10.5281/zenodo.5143773"});
CREATE (mcro_clipViTbigG14laion2B_CitationScalingOpenCLIP:CitationInformationSection {id: "mcro:clipViTbigG14laion2B-CitationScalingOpenCLIP", hasTextValue: "@articlecherti2022reproducible,
  title=Reproducible scaling laws for contrastive language-image learning,
  author=Cherti, Mehdi and Beaumont, Romain and Wightman, Ross and Wortsman, Mitchell and Ilharco, Gabriel and Gordon, Cade and Schuhmann, Christoph and Schmidt, Ludwig and Jitsev, Jenia,
  journal=arXiv preprint arXiv:2212.07143,
  year=2022"});
CREATE (mcro_clipViTbigG14laion2B_UseCase:UseCaseInformationSection {id: "mcro:clipViTbigG14laion2B-UseCase"});
CREATE (mcro_clipViTbigG14laion2B_ModelParameter:ModelParameterSection {id: "mcro:clipViTbigG14laion2B-ModelParameter"});
CREATE (mcro_clipViTbigG14laion2B_Dataset:DatasetInformationSection {id: "mcro:clipViTbigG14laion2B-Dataset"});
CREATE (mcro_clipViTbigG14laion2B_QuantativeAnalysis:QuantativeAnalysisSection {id: "mcro:clipViTbigG14laion2B-QuantativeAnalysis"});
CREATE (mcro_allMiniLML12v2:Model {id: "mcro:allMiniLML12v2"});
CREATE (mcro_allMiniLML12v2_UseCaseInformationSection:UseCaseInformationSection {id: "mcro:allMiniLML12v2-UseCaseInformationSection", hasTextValue: "Our model is intented to be used as a sentence and short paragraph encoder. Given an input text, it ouptuts a vector which captures 
the semantic information. The sentence vector may be used for information retrieval, clustering or sentence similarity tasks."});
CREATE (mcro_allMiniLML12v2_TrainingDataInformationSection:TrainingDataInformationSection {id: "mcro:allMiniLML12v2-TrainingDataInformationSection", hasTextValue: "We use the concatenation from multiple datasets to fine-tune our model. The total number of sentence pairs is above 1 billion sentences.
We sampled each dataset given a weighted probability which configuration is detailed in the `data_config.json` file."});
CREATE (mcro_allMiniLML12v2_ModelArchitectureInformationSection:ModelArchitectureInformationSection {id: "mcro:allMiniLML12v2-ModelArchitectureInformationSection", hasTextValue: "We used the pretrained [`microsoft/MiniLM-L12-H384-uncased`](https://huggingface.co/microsoft/MiniLM-L12-H384-uncased) model and fine-tuned in on a 
1B sentence pairs dataset."});
CREATE (mcro_tabletransformer:Model {id: "mcro:tabletransformer"});
CREATE (mcro_tabletransformer_ModelDetail:ModelDetailSection {id: "mcro:tabletransformer-ModelDetail"});
CREATE (mcro_tabletransformer_Citation:CitationInformationSection {id: "mcro:tabletransformer-Citation", hasTextValue: "PubTables-1M: Towards Comprehensive Table Extraction From Unstructured Documents"});
CREATE (mcro_tabletransformer_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:tabletransformer-ModelArchitecture", hasTextValue: "Transformer-based object detection model"});
CREATE (mcro_tabletransformer_UseCase:UseCaseInformationSection {id: "mcro:tabletransformer-UseCase", hasTextValue: "detecting tables in documents"});
CREATE (mcro_t5small:Model {id: "mcro:t5small"});
CREATE (mcro_t5small_ModelDetails:ModelDetailSection {id: "mcro:t5small-ModelDetails"});
CREATE (mcro_t5small_License:LicenseInformationSection {id: "mcro:t5small-License", hasTextValue: "Apache 2.0"});
CREATE (mcro_t5small_Citation:CitationInformationSection {id: "mcro:t5small-Citation"});
CREATE (mcro_t5small_UseCase:UseCaseInformationSection {id: "mcro:t5small-UseCase"});
CREATE (mcro_t5small_Dataset:DatasetInformationSection {id: "mcro:t5small-Dataset", hasTextValue: "Colossal Clean Crawled Corpus (C4)"});
CREATE (mcro_t5small_Architecture:ModelArchitectureInformationSection {id: "mcro:t5small-Architecture", hasTextValue: "Language model"});
CREATE (mcro_albertbasev2:Model {id: "mcro:albertbasev2"});
CREATE (mcro_albertbasev2_ModelDetail:ModelDetailSection {id: "mcro:albertbasev2-ModelDetail"});
CREATE (mcro_albertbasev2_Citation:CitationInformationSection {id: "mcro:albertbasev2-Citation", hasTextValue: "@articleDBLP:journals/corr/abs-1909-11942,
  author    = Zhenzhong Lan and
               Mingda Chen and
               Sebastian Goodman and
               Kevin Gimpel and
               Piyush Sharma and
               Radu Soricut,
  title     = ALBERT: A Lite BERT for Self-supervised Learning of Language
               Representations,
  journal   = CoRR,
  volume    = abs/1909.11942,
  year      = 2019,
  url       = http://arxiv.org/abs/1909.11942,
  archivePrefix = arXiv,
  eprint    = 1909.11942,
  timestamp = Fri, 27 Sep 2019 13:04:21 +0200,
  biburl    = https://dblp.org/rec/journals/corr/abs-1909-11942.bib,
  bibsource = dblp computer science bibliography, https://dblp.org"});
CREATE (mcro_albertbasev2_License:LicenseInformationSection {id: "mcro:albertbasev2-License"});
CREATE (mcro_albertbasev2_Architecture:ModelArchitectureInformationSection {id: "mcro:albertbasev2-Architecture"});
CREATE (mcro_albertbasev2_UseCase:UseCaseInformationSection {id: "mcro:albertbasev2-UseCase"});
CREATE (mcro_albertbasev2_TrainingData:TrainingDataInformationSection {id: "mcro:albertbasev2-TrainingData"});
CREATE (mcro_albertbasev2_Limitation:LimitationInformationSection {id: "mcro:albertbasev2-Limitation"});
CREATE (mcro_distilgpt2:Model {id: "mcro:distilgpt2"});
CREATE (mcro_distilgpt2_ModelDetail:ModelDetailSection {id: "mcro:distilgpt2-ModelDetail"});
CREATE (mcro_distilgpt2_License:LicenseInformationSection {id: "mcro:distilgpt2-License", hasTextValue: "Apache 2.0"});
CREATE (mcro_distilgpt2_Architecture:ModelArchitectureInformationSection {id: "mcro:distilgpt2-Architecture", hasTextValue: "Transformer-based Language Model"});
CREATE (mcro_distilgpt2_IntendedUse:UseCaseInformationSection {id: "mcro:distilgpt2-IntendedUse"});
CREATE (mcro_distilgpt2_Limitation:LimitationInformationSection {id: "mcro:distilgpt2-Limitation"});
CREATE (mcro_distilgpt2_TrainingData:DatasetInformationSection {id: "mcro:distilgpt2-TrainingData"});
CREATE (mcro_distilgpt2_TrainingProcedure:IAO_0000314 {id: "mcro:distilgpt2-TrainingProcedure"});
CREATE (mcro_distilgpt2_EvaluationResult:IAO_0000314 {id: "mcro:distilgpt2-EvaluationResult"});
CREATE (mcro_distilgpt2_EnvironmentalImpact:IAO_0000314 {id: "mcro:distilgpt2-EnvironmentalImpact"});
CREATE (mcro_distilgpt2_Citation:CitationInformationSection {id: "mcro:distilgpt2-Citation"});
CREATE (mcro_forcedalignerwithhuggingfacectcmodels:Model {id: "mcro:forcedalignerwithhuggingfacectcmodels"});
CREATE (mcro_forcedalignerwithhuggingfacectcmodels_intendedusecase:UseCaseInformationSection {id: "mcro:forcedalignerwithhuggingfacectcmodels-intendedusecase", hasTextValue: "This Python package provides an efficient way to perform forced alignment between text and audio using Hugging Face's pretrained models. it also features an improved implementation to use much less memory than TorchAudio forced alignment API."});
CREATE (mcro_forcedalignerwithhuggingfacectcmodels_modelarchitecture:ModelArchitectureInformationSection {id: "mcro:forcedalignerwithhuggingfacectcmodels-modelarchitecture", hasTextValue: "The model checkpoint uploaded here is a conversion from torchaudio to HF Transformers for the MMS-300M checkpoint trained on forced alignment dataset"});
CREATE (mcro_mixedbreadaimxbaiembedlargev1:Model {id: "mcro:mixedbreadaimxbaiembedlargev1"});
CREATE (mcro_mixedbreadaimxbaiembedlargev1_License:LicenseInformationSection {id: "mcro:mixedbreadaimxbaiembedlargev1-License", hasTextValue: "Apache 2.0"});
CREATE (mcro_mixedbreadaimxbaiembedlargev1_Citation1:CitationInformationSection {id: "mcro:mixedbreadaimxbaiembedlargev1-Citation1", hasTextValue: "@onlineemb2024mxbai,
  title=Open Source Strikes Bread - New Fluffy Embeddings Model,
  author=Sean Lee and Aamir Shakir and Darius Koenig and Julius Lipp,
  year=2024,
  url=https://www.mixedbread.ai/blog/mxbai-embed-large-v1,"});
CREATE (mcro_mixedbreadaimxbaiembedlargev1_Citation2:CitationInformationSection {id: "mcro:mixedbreadaimxbaiembedlargev1-Citation2", hasTextValue: "@articleli2023angle,
  title=AnglE-optimized Text Embeddings,
  author=Li, Xianming and Li, Jing,
  journal=arXiv preprint arXiv:2309.12871,
  year=2023"});
CREATE (mcro_mixedbreadaimxbaiembedlargev1_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:mixedbreadaimxbaiembedlargev1-ModelArchitecture"});
CREATE (mcro_mixedbreadaimxbaiembedlargev1_Dataset:DatasetInformationSection {id: "mcro:mixedbreadaimxbaiembedlargev1-Dataset"});
CREATE (mcro_mixedbreadaimxbaiembedlargev1_UseCase:UseCaseInformationSection {id: "mcro:mixedbreadaimxbaiembedlargev1-UseCase"});
CREATE (mcro_facebookdinov2small:Model {id: "mcro:facebookdinov2small"});
CREATE (mcro_facebookdinov2small_ModelDetail:ModelDetailSection {id: "mcro:facebookdinov2small-ModelDetail"});
CREATE (mcro_facebookdinov2small_Citation:CitationInformationSection {id: "mcro:facebookdinov2small-Citation", hasTextValue: "miscoquab2023dinov2,
      title=DINOv2: Learning Robust Visual Features without Supervision, 
      author=Maxime Oquab and Timothée Darcet and Théo Moutakanni and Huy Vo and Marc Szafraniec and Vasil Khalidov and Pierre Fernandez and Daniel Haziza and Francisco Massa and Alaaeldin El-Nouby and Mahmoud Assran and Nicolas Ballas and Wojciech Galuba and Russell Howes and Po-Yao Huang and Shang-Wen Li and Ishan Misra and Michael Rabbat and Vasu Sharma and Gabriel Synnaeve and Hu Xu and Hervé Jegou and Julien Mairal and Patrick Labatut and Armand Joulin and Piotr Bojanowski,
      year=2023,
      eprint=2304.07193,
      archivePrefix=arXiv,
      primaryClass=cs.CV"});
CREATE (mcro_facebookdinov2small_License:LicenseInformationSection {id: "mcro:facebookdinov2small-License"});
CREATE (mcro_facebookdinov2small_Architecture:ModelArchitectureInformationSection {id: "mcro:facebookdinov2small-Architecture", hasTextValue: "Vision Transformer (ViT)"});
CREATE (mcro_facebookdinov2small_UseCase:UseCaseInformationSection {id: "mcro:facebookdinov2small-UseCase"});
CREATE (mcro_efficientnet_b3ra2in1k:Model {id: "mcro:efficientnet_b3ra2in1k"});
CREATE (mcro_efficientnet_b3ra2in1k_Dataset:DatasetInformationSection {id: "mcro:efficientnet_b3ra2in1k-Dataset", hasTextValue: "ImageNet-1k"});
CREATE (mcro_efficientnet_b3ra2in1k_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:efficientnet_b3ra2in1k-ModelArchitecture", hasTextValue: "EfficientNet"});
CREATE (mcro_efficientnet_b3ra2in1k_Citation_1:CitationInformationSection {id: "mcro:efficientnet_b3ra2in1k-Citation-1", hasTextValue: "@inproceedingstan2019efficientnet,
  title=Efficientnet: Rethinking model scaling for convolutional neural networks,
  author=Tan, Mingxing and Le, Quoc,
  booktitle=International conference on machine learning,
  pages=6105--6114,
  year=2019,
  organization=PMLR"});
CREATE (mcro_efficientnet_b3ra2in1k_Citation_2:CitationInformationSection {id: "mcro:efficientnet_b3ra2in1k-Citation-2", hasTextValue: "@miscrw2019timm,
  author = Ross Wightman,
  title = PyTorch Image Models,
  year = 2019,
  publisher = GitHub,
  journal = GitHub repository,
  doi = 10.5281/zenodo.4414861,
  howpublished = https://github.com/huggingface/pytorch-image-models"});
CREATE (mcro_efficientnet_b3ra2in1k_Citation_3:CitationInformationSection {id: "mcro:efficientnet_b3ra2in1k-Citation-3", hasTextValue: "@inproceedingswightman2021resnet,
  title=ResNet strikes back: An improved training procedure in timm,
  author=Wightman, Ross and Touvron, Hugo and Jegou, Herve,
  booktitle=NeurIPS 2021 Workshop on ImageNet: Past, Present, and Future"});
CREATE (mcro_efficientnet_b3ra2in1k_UseCase:UseCaseInformationSection {id: "mcro:efficientnet_b3ra2in1k-UseCase", hasTextValue: "Image classification / feature backbone"});
CREATE (mcro_Qwen25VL7BInstruct:Model {id: "mcro:Qwen25VL7BInstruct"});
CREATE (mcro_Qwen25VL7BInstruct_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:Qwen25VL7BInstruct-ModelArchitecture"});
CREATE (mcro_Qwen25VL7BInstruct_Citation:CitationInformationSection {id: "mcro:Qwen25VL7BInstruct-Citation", hasTextValue: "@miscqwen2.5-VL,
    title = Qwen2.5-VL,
    url = https://qwenlm.github.io/blog/qwen2.5-vl/,
    author = Qwen Team,
    month = January,
    year = 2025


@articleQwen2VL,
  title=Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution,
  author=Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang,
  journal=arXiv preprint arXiv:2409.12191,
  year=2024


@articleQwen-VL,
  title=Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond,
  author=Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren,
  journal=arXiv preprint arXiv:2308.12966,
  year=2023"});
CREATE (mcro_Qwen25VL7BInstruct_IntendedUseCase:UseCaseInformationSection {id: "mcro:Qwen25VL7BInstruct-IntendedUseCase"});
CREATE (mcro_Qwen25VL7BInstruct_Dataset:DatasetInformationSection {id: "mcro:Qwen25VL7BInstruct-Dataset"});
CREATE (mcro_gemma3:Model {id: "mcro:gemma3"});
CREATE (mcro_gemma3_ModelDetailSection:ModelDetailSection {id: "mcro:gemma3-ModelDetailSection"});
CREATE (mcro_gemma3_CitationSection:CitationInformationSection {id: "mcro:gemma3-CitationSection", hasTextValue: "@articlegemma_2025,
    title=Gemma 3,
    url=https://goo.gle/Gemma3Report,
    publisher=Kaggle,
    author=Gemma Team,
    year=2025"});
CREATE (mcro_gemma3_ModelParameterSection:ModelParameterSection {id: "mcro:gemma3-ModelParameterSection"});
CREATE (mcro_gemma3_DatasetSection:DatasetInformationSection {id: "mcro:gemma3-DatasetSection"});
CREATE (mcro_gemma3_ModelArchitectureSection:ModelArchitectureInformationSection {id: "mcro:gemma3-ModelArchitectureSection"});
CREATE (mcro_gemma3_QuantativeAnalysisSection:QuantativeAnalysisSection {id: "mcro:gemma3-QuantativeAnalysisSection"});
CREATE (mcro_gemma3_ConsiderationSection:ConsiderationInformationSection {id: "mcro:gemma3-ConsiderationSection"});
CREATE (mcro_gemma3_UseCaseSection:UseCaseInformationSection {id: "mcro:gemma3-UseCaseSection"});
CREATE (mcro_gemma3_ModelCardReport:ModelCardReport {id: "mcro:gemma3-ModelCardReport"});
CREATE (mcro_sentencetransformersparaphrasemultilingualmpnetbasv2:Model {id: "mcro:sentencetransformersparaphrasemultilingualmpnetbasv2"});
CREATE (mcro_sentencetransformersparaphrasemultilingualmpnetbasv2_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:sentencetransformersparaphrasemultilingualmpnetbasv2-ModelArchitecture", hasTextValue: "SentenceTransformer(
  (0): Transformer('max_seq_length': 128, 'do_lower_case': False) with Transformer model: XLMRobertaModel 
  (1): Pooling('word_embedding_dimension': 768, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False)
)"});
CREATE (mcro_sentencetransformersparaphrasemultilingualmpnetbasv2_Citation:CitationInformationSection {id: "mcro:sentencetransformersparaphrasemultilingualmpnetbasv2-Citation", hasTextValue: "@inproceedingsreimers-2019-sentence-bert,
    title = \"Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks\",
    author = \"Reimers, Nils and Gurevych, Iryna\",
    booktitle = \"Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing\",
    month = \"11\",
    year = \"2019\",
    publisher = \"Association for Computational Linguistics\",
    url = \"http://arxiv.org/abs/1908.10084\","});
CREATE (mcro_sentencetransformersparaphrasemultilingualmpnetbasv2_UseCase:UseCaseInformationSection {id: "mcro:sentencetransformersparaphrasemultilingualmpnetbasv2-UseCase", hasTextValue: "It maps sentences & paragraphs to a 768 dimensional dense vector space and can be used for tasks like clustering or semantic search."});
CREATE (mcro_Salesforceblipbootstrapping:Model {id: "mcro:Salesforceblipbootstrapping"});
CREATE (mcro_Salesforceblipbootstrapping_Citation:CitationInformationSection {id: "mcro:Salesforceblipbootstrapping-Citation", hasTextValue: "@mischttps://doi.org/10.48550/arxiv.2201.12086,
  doi = 10.48550/ARXIV.2201.12086,
  
  url = https://arxiv.org/abs/2201.12086,
  
  author = Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven,
  
  keywords = Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences,
  
  title = BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation,
  
  publisher = arXiv,
  
  year = 2022,
  
  copyright = Creative Commons Attribution 4.0 International"});
CREATE (mcro_Salesforceblipbootstrapping_UseCase:UseCaseInformationSection {id: "mcro:Salesforceblipbootstrapping-UseCase", hasTextValue: "You can use this model for conditional and un-conditional image captioning"});
CREATE (mcro_Salesforceblipbootstrapping_License:LicenseInformationSection {id: "mcro:Salesforceblipbootstrapping-License", hasTextValue: "This release is for research purposes only in support of an academic paper. Our models, datasets, and code are not specifically designed or evaluated for all downstream purposes. We strongly recommend users evaluate and address potential concerns related to accuracy, safety, and fairness before deploying this model. We encourage users to consider the common limitations of AI, comply with applicable laws, and leverage best practices when selecting use cases, particularly for high-risk scenarios where errors or misuse could significantly impact people’s lives, rights, or safety. For further guidance on use cases, refer to our AUP and AI AUP."});
CREATE (mcro_Salesforceblipbootstrapping_Architecture:ModelArchitectureInformationSection {id: "mcro:Salesforceblipbootstrapping-Architecture", hasTextValue: "base architecture (with ViT base backbone)"});
CREATE (mcro_Salesforceblipbootstrapping_Consideration:ConsiderationInformationSection {id: "mcro:Salesforceblipbootstrapping-Consideration", hasTextValue: "We strongly recommend users evaluate and address potential concerns related to accuracy, safety, and fairness before deploying this model. We encourage users to consider the common limitations of AI, comply with applicable laws, and leverage best practices when selecting use cases, particularly for high-risk scenarios where errors or misuse could significantly impact people’s lives, rights, or safety."});
CREATE (mcro_cocoDataset:DatasetInformationSection {id: "mcro:cocoDataset", hasTextValue: "COCO dataset"});
CREATE (mcro_flux1dev:Model {id: "mcro:flux1dev"});
CREATE (mcro_flux1dev_License:LicenseInformationSection {id: "mcro:flux1dev-License", hasTextValue: "`FLUX.1 [dev]` Non-Commercial License"});
CREATE (mcro_flux1dev_Limitation:LimitationInformationSection {id: "mcro:flux1dev-Limitation", hasTextValue: "Prompt following is heavily influenced by the prompting-style."});
CREATE (mcro_flux1dev_UseCase:UseCaseInformationSection {id: "mcro:flux1dev-UseCase", hasTextValue: "personal, scientific, and commercial purposes"});
CREATE (mcro_flux1dev_OutOfScopeUse:OutOfScopeUseCaseSectionInformation {id: "mcro:flux1dev-OutOfScopeUse", hasTextValue: "Generating or facilitating large-scale disinformation campaigns."});
CREATE (mcro_flux1dev_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:flux1dev-ModelArchitecture", hasTextValue: "12 billion parameter rectified flow transformer"});
CREATE (mcro_unik3d:Model {id: "mcro:unik3d"});
CREATE (mcro_unik3d_Library:LibraryInformationSection {id: "mcro:unik3d-Library", hasTextValue: "https://github.com/lpiccinelli-eth/UniK3D"});
CREATE (mcro_unik3d_Documentation:DocumentationInformationSection {id: "mcro:unik3d-Documentation", hasTextValue: "[More Information Needed]"});
CREATE (mcro_llama32:Model {id: "mcro:llama32"});
CREATE (mcro_llama32_modelDetail:ModelDetailSection {id: "mcro:llama32-modelDetail"});
CREATE (mcro_llama32_license:LicenseInformationSection {id: "mcro:llama32-license", hasTextValue: "Use of Llama 3.2 is governed by the [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) (a custom, commercial license agreement)."});
CREATE (mcro_llama32_citation:CitationInformationSection {id: "mcro:llama32-citation", hasTextValue: "Instructions on how to provide feedback or comments on the model can be found in the Llama Models [README](https://github.com/meta-llama/llama-models/blob/main/README.md). For more technical information about generation parameters and recipes for how to use Llama 3.2 in applications, please go [here](https://github.com/meta-llama/llama-recipes)."});
CREATE (mcro_llama32_architecture:ModelArchitectureInformationSection {id: "mcro:llama32-architecture", hasTextValue: "Llama 3.2 is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use supervised fine-tuning (SFT) and reinforcement learning with human feedback (RLHF) to align with human preferences for helpfulness and safety."});
CREATE (mcro_llama32_usecase:UseCaseInformationSection {id: "mcro:llama32-usecase", hasTextValue: "Llama 3.2 is intended for commercial and research use in multiple languages. Instruction tuned text only models are intended for assistant-like chat and agentic applications like knowledge retrieval and summarization, mobile AI powered writing assistants and query and prompt rewriting. Pretrained models can be adapted for a variety of additional natural language generation tasks. Similarly, quantized models can be adapted for a variety of on-device use-cases with limited compute resources."});
CREATE (mcro_llama32_outofscope:OutOfScopeUseCaseSectionInformation {id: "mcro:llama32-outofscope", hasTextValue: "Use in any manner that violates applicable laws or regulations (including trade compliance laws). Use in any other way that is prohibited by the Acceptable Use Policy and Llama 3.2 Community License. Use in languages beyond those explicitly referenced as supported in this model card."});
CREATE (mcro_llama32_parameter:ModelParameterSection {id: "mcro:llama32-parameter"});
CREATE (mcro_llama32_trainingdata:TrainingDataInformationSection {id: "mcro:llama32-trainingdata", hasTextValue: "Llama 3.2 was pretrained on up to 9 trillion tokens of data from publicly available sources. For the 1B and 3B Llama 3.2 models, we incorporated logits from the Llama 3.1 8B and 70B models into the pretraining stage of the model development, where outputs (logits) from these larger models were used as token-level targets. Knowledge distillation was used after pruning to recover performance. In post-training we used a similar recipe as Llama 3.1 and produced final chat models by doing several rounds of alignment on top of the pre-trained model. Each round involved Supervised Fine-Tuning (SFT), Rejection Sampling (RS), and Direct Preference Optimization (DPO)."});
CREATE (mcro_bertmultilingualbasemodeluncased:IAO_0000301:Model {id: "mcro:bertmultilingualbasemodeluncased"});
CREATE (mcro_bertmultilingualbasemodeluncased_Citation:CitationInformationSection {id: "mcro:bertmultilingualbasemodeluncased-Citation", hasTextValue: "@articleDBLP:journals/corr/abs-1810-04805,
  author    = Jacob Devlin and
               Ming-Wei Chang and
               Kenton Lee and
               Kristina Toutanova,
  title     = BERT: Pre-training of Deep Bidirectional Transformers for Language
               Understanding,
  journal   = CoRR,
  volume    = abs/1810.04805,
  year      = 2018,
  url       = http://arxiv.org/abs/1810.04805,
  archivePrefix = arXiv,
  eprint    = 1810.04805,
  timestamp = Tue, 30 Oct 2018 20:39:56 +0100,
  biburl    = https://dblp.org/rec/journals/corr/abs-1810-04805.bib,
  bibsource = dblp computer science bibliography, https://dblp.org"});
CREATE (mcro_bertmultilingualbasemodeluncased_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:bertmultilingualbasemodeluncased-ModelArchitecture", hasTextValue: "BERT is a transformers model pretrained on a large corpus of multilingual data in a self-supervised fashion. This means
it was pretrained on the raw texts only, with no humans labelling them in any way (which is why it can use lots of
publicly available data) with an automatic process to generate inputs and labels from those texts. More precisely, it
was pretrained with two objectives:

- Masked language modeling (MLM): taking a sentence, the model randomly masks 15% of the words in the input then run
  the entire masked sentence through the model and has to predict the masked words. This is different from traditional
  recurrent neural networks (RNNs) that usually see the words one after the other, or from autoregressive models like
  GPT which internally mask the future tokens. It allows the model to learn a bidirectional representation of the
  sentence.
- Next sentence prediction (NSP): the models concatenates two masked sentences as inputs during pretraining. Sometimes
  they correspond to sentences that were next to each other in the original text, sometimes not. The model then has to
  predict if the two sentences were following each other or not.

This way, the model learns an inner representation of the languages in the training set that can then be used to
extract features useful for downstream tasks: if you have a dataset of labeled sentences for instance, you can train a
standard classifier using the features produced by the BERT model as inputs."});
CREATE (mcro_bertmultilingualbasemodeluncased_IntendedUseCase:UseCaseInformationSection {id: "mcro:bertmultilingualbasemodeluncased-IntendedUseCase", hasTextValue: "You can use the raw model for either masked language modeling or next sentence prediction, but it's mostly intended to
be fine-tuned on a downstream task. See the [model hub](https://huggingface.co/models?filter=bert) to look for
fine-tuned versions on a task that interests you.

Note that this model is primarily aimed at being fine-tuned on tasks that use the whole sentence (potentially masked)
to make decisions, such as sequence classification, token classification or question answering. For tasks such as text
generation you should look at model like GPT2."});
CREATE (mcro_bertmultilingualbasemodeluncased_Dataset:DatasetInformationSection {id: "mcro:bertmultilingualbasemodeluncased-Dataset", hasTextValue: "The BERT model was pretrained on the 102 languages with the largest Wikipedias. You can find the complete list
[here](https://github.com/google-research/bert/blob/master/multilingual.md#list-of-languages)."});
CREATE (mcro_envbreaker:Model {id: "mcro:envbreaker", hasTextValue: "We log statistics to see if any envs are breaking"});
CREATE (mcro_Salesforceblipimagecaptioninglarge:Model {id: "mcro:Salesforceblipimagecaptioninglarge"});
CREATE (mcro_Salesforceblipimagecaptioninglarge_Citation:CitationInformationSection {id: "mcro:Salesforceblipimagecaptioninglarge-Citation", hasTextValue: "@mischttps://doi.org/10.48550/arxiv.2201.12086,
  doi = 10.48550/ARXIV.2201.12086,
  
  url = https://arxiv.org/abs/2201.12086,
  
  author = Li, Junnan and Li, Dongxu and Xiong, Caiming and Hoi, Steven,
  
  keywords = Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences,
  
  title = BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation,
  
  publisher = arXiv,
  
  year = 2022,
  
  copyright = Creative Commons Attribution 4.0 International"});
CREATE (mcro_Salesforceblipimagecaptioninglarge_IntendedUseCase:UseCaseInformationSection {id: "mcro:Salesforceblipimagecaptioninglarge-IntendedUseCase", hasTextValue: "You can use this model for conditional and un-conditional image captioning"});
CREATE (mcro_Salesforceblipimagecaptioninglarge_Considerations:ConsiderationInformationSection {id: "mcro:Salesforceblipimagecaptioninglarge-Considerations", hasTextValue: "This release is for research purposes only in support of an academic paper. Our models, datasets, and code are not specifically designed or evaluated for all downstream purposes. We strongly recommend users evaluate and address potential concerns related to accuracy, safety, and fairness before deploying this model. We encourage users to consider the common limitations of AI, comply with applicable laws, and leverage best practices when selecting use cases, particularly for high-risk scenarios where errors or misuse could significantly impact people’s lives, rights, or safety. For further guidance on use cases, refer to our AUP and AI AUP."});
CREATE (mcro_Salesforceblipimagecaptioninglarge_Architecture:ModelArchitectureInformationSection {id: "mcro:Salesforceblipimagecaptioninglarge-Architecture", hasTextValue: "base architecture (with ViT large backbone)"});
CREATE (mcro_Salesforceblipimagecaptioninglarge_Dataset:DatasetInformationSection {id: "mcro:Salesforceblipimagecaptioninglarge-Dataset", hasTextValue: "COCO dataset"});
CREATE (mcro_crossencodermsmarcoMiniLML6v2:Model {id: "mcro:crossencodermsmarcoMiniLML6v2"});
CREATE (mcro_crossencodermsmarcoMiniLML6v2_UseCase:UseCaseInformationSection {id: "mcro:crossencodermsmarcoMiniLML6v2-UseCase", hasTextValue: "This model was trained on the [MS Marco Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking) task.

The model can be used for Information Retrieval: Given a query, encode the query will all possible passages (e.g. retrieved with ElasticSearch). Then sort the passages in a decreasing order. See [SBERT.net Retrieve & Re-rank](https://www.sbert.net/examples/applications/retrieve_rerank/README.html) for more details. The training code is available here: [SBERT.net Training MS Marco](https://github.com/UKPLab/sentence-transformers/tree/master/examples/training/ms_marco)"});
CREATE (mcro_crossencodermsmarcoMiniLML6v2_Performance:PerformanceMetricInformationSection {id: "mcro:crossencodermsmarcoMiniLML6v2-Performance", hasTextValue: "In the following table, we provide various pre-trained Cross-Encoders together with their performance on the [TREC Deep Learning 2019](https://microsoft.github.io/TREC-2019-Deep-Learning/) and the [MS Marco Passage Reranking](https://github.com/microsoft/MSMARCO-Passage-Ranking/) dataset."});
CREATE (mcro_sdxl10base:Model {id: "mcro:sdxl10base"});
CREATE (mcro_sdxl10base_ModelDetail:ModelDetailSection {id: "mcro:sdxl10base-ModelDetail"});
CREATE (mcro_sdxl10base_License:LicenseInformationSection {id: "mcro:sdxl10base-License", hasTextValue: "CreativeML Open RAIL++-M License"});
CREATE (mcro_sdxl10base_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:sdxl10base-ModelArchitecture", hasTextValue: "Diffusion-based text-to-image generative model"});
CREATE (mcro_sdxl10base_Citation:CitationInformationSection {id: "mcro:sdxl10base-Citation", hasTextValue: "SDXL report on arXiv"});
CREATE (mcro_sdxl10base_UseCase:UseCaseInformationSection {id: "mcro:sdxl10base-UseCase", hasTextValue: "Probing and understanding the limitations and biases of generative models."});
CREATE (mcro_sdxl10base_Consideration:ConsiderationInformationSection {id: "mcro:sdxl10base-Consideration"});
CREATE (mcro_sdxl10base_Limitation:LimitationInformationSection {id: "mcro:sdxl10base-Limitation"});
CREATE (mcro_sdxl10base_Risk:RiskInformationSection {id: "mcro:sdxl10base-Risk"});
CREATE (mcro_sdxl10base_OutOfScopeUseCase:OutOfScopeUseCaseSectionInformation {id: "mcro:sdxl10base-OutOfScopeUseCase"});
CREATE (mcro_sdxl10base_Dataset:DatasetInformationSection {id: "mcro:sdxl10base-Dataset"});
CREATE (mcro_whisper:Model {id: "mcro:whisper"});
CREATE (mcro_whisper_ModelDetail:ModelDetailSection {id: "mcro:whisper-ModelDetail"});
CREATE (mcro_whisper_Citation:CitationInformationSection {id: "mcro:whisper-Citation", hasTextValue: "@miscradford2022whisper,
  doi = 10.48550/ARXIV.2212.04356,
  url = https://arxiv.org/abs/2212.04356,
  author = Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya,
  title = Robust Speech Recognition via Large-Scale Weak Supervision,
  publisher = arXiv,
  year = 2022,
  copyright = arXiv.org perpetual, non-exclusive license"});
CREATE (mcro_whisper_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:whisper-ModelArchitecture", hasTextValue: "Transformer based encoder-decoder model"});
CREATE (mcro_whisper_IntendedUseCase:UseCaseInformationSection {id: "mcro:whisper-IntendedUseCase", hasTextValue: "ASR solution"});
CREATE (mcro_whisper_TrainingData:TrainingDataInformationSection {id: "mcro:whisper-TrainingData", hasTextValue: "680,000 hours of audio and the corresponding transcripts collected from the internet"});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53dutch:Model {id: "mcro:jonatasgrosmanwav2vec2largexlsr53dutch"});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53dutch_Citation:CitationInformationSection {id: "mcro:jonatasgrosmanwav2vec2largexlsr53dutch-Citation", hasTextValue: "@miscgrosman2021xlsr53-large-dutch,
  title=Fine-tuned XLSR-53 large model for speech recognition in Dutch,
  author=Grosman, Jonatas,
  howpublished=https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-dutch,
  year=2021"});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53dutch_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:jonatasgrosmanwav2vec2largexlsr53dutch-ModelArchitecture", hasTextValue: "XLSR-53 large"});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53dutch_Dataset:DatasetInformationSection {id: "mcro:jonatasgrosmanwav2vec2largexlsr53dutch-Dataset", hasTextValue: "Common Voice 6.1"});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53dutch_Dataset2:DatasetInformationSection {id: "mcro:jonatasgrosmanwav2vec2largexlsr53dutch-Dataset2", hasTextValue: "CSS10"});
CREATE (mcro_jonatasgrosmanwav2vec2largexlsr53dutch_UseCase:UseCaseInformationSection {id: "mcro:jonatasgrosmanwav2vec2largexlsr53dutch-UseCase", hasTextValue: "speech recognition in Dutch"});
CREATE (mcro_visiontransformerbase_Citation1:CitationInformationSection {id: "mcro:visiontransformerbase-Citation1", hasTextValue: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"});
CREATE (mcro_visiontransformerbase_Citation2:CitationInformationSection {id: "mcro:visiontransformerbase-Citation2", hasTextValue: "Visual Transformers: Token-based Image Representation and Processing for Computer Vision"});
CREATE (mcro_visiontransformerbase_Citation3:CitationInformationSection {id: "mcro:visiontransformerbase-Citation3", hasTextValue: "Imagenet: A large-scale hierarchical image database"});
CREATE (mcro_clinicalBERT_Bio_ClinicalBERTModel:Model {id: "mcro:clinicalBERT-Bio+ClinicalBERTModel"});
CREATE (mcro_clinicalBERT_Bio_ClinicalBERTModel_DatasetInfo:DatasetInformationSection {id: "mcro:clinicalBERT-Bio+ClinicalBERTModel-DatasetInfo"});
CREATE (mcro_clinicalBERT_Bio_ClinicalBERTModel_ArchInfo:ModelArchitectureInformationSection {id: "mcro:clinicalBERT-Bio+ClinicalBERTModel-ArchInfo"});
CREATE (mcro_clinicalBERT_Bio_ClinicalBERTModel_CitationInfo:CitationInformationSection {id: "mcro:clinicalBERT-Bio+ClinicalBERTModel-CitationInfo"});
CREATE (mcro_clinicalBERT_Bio_ClinicalBERTModel_UseCaseInfo:UseCaseInformationSection {id: "mcro:clinicalBERT-Bio+ClinicalBERTModel-UseCaseInfo"});
CREATE (mcro_clinicalBERT_Bio_ClinicalBERTModel_TrainingDataInfo:TrainingDataInformationSection {id: "mcro:clinicalBERT-Bio+ClinicalBERTModel-TrainingDataInfo"});
CREATE (mcro_FlagEmbedding_CitationInformationSection:CitationInformationSection {id: "mcro:FlagEmbedding-CitationInformationSection", hasTextValue: "@miscbge_embedding,
      title=C-Pack: Packaged Resources To Advance General Chinese Embedding, 
      author=Shitao Xiao and Zheng Liu and Peitian Zhang and Niklas Muennighoff,
      year=2023,
      eprint=2309.07597,
      archivePrefix=arXiv,
      primaryClass=cs.CL"});
CREATE (mcro_FlagEmbedding_LicenseInformationSection:LicenseInformationSection {id: "mcro:FlagEmbedding-LicenseInformationSection", hasTextValue: "FlagEmbedding is licensed under the [MIT License](https://github.com/FlagOpen/FlagEmbedding/blob/master/LICENSE). The released models can be used for commercial purposes free of charge."});
CREATE (mcro_facebookbartbase:Model {id: "mcro:facebookbartbase"});
CREATE (mcro_facebookbartbase_ModelDetail:ModelDetailSection {id: "mcro:facebookbartbase-ModelDetail"});
CREATE (mcro_facebookbartbase_Citation:CitationInformationSection {id: "mcro:facebookbartbase-Citation", hasTextValue: "@articleDBLP:journals/corr/abs-1910-13461,
  author    = Mike Lewis and
               Yinhan Liu and
               Naman Goyal and
               Marjan Ghazvininejad and
               Abdelrahman Mohamed and
               Omer Levy and
               Veselin Stoyanov and
               Luke Zettlemoyer,
  title     = BART: Denoising Sequence-to-Sequence Pre-training for Natural Language
               Generation, Translation, and Comprehension,
  journal   = CoRR,
  volume    = abs/1910.13461,
  year      = 2019,
  url       = http://arxiv.org/abs/1910.13461,
  eprinttype = arXiv,
  eprint    = 1910.13461,
  timestamp = Thu, 31 Oct 2019 14:02:26 +0100,
  biburl    = https://dblp.org/rec/journals/corr/abs-1910-13461.bib,
  bibsource = dblp computer science bibliography, https://dblp.org"});
CREATE (mcro_facebookbartbase_ModelArchitecture:ModelArchitectureInformationSection {id: "mcro:facebookbartbase-ModelArchitecture", hasTextValue: "BART is a transformer encoder-decoder (seq2seq) model with a bidirectional (BERT-like) encoder and an autoregressive (GPT-like) decoder."});
CREATE (mcro_facebookbartbase_UseCase:UseCaseInformationSection {id: "mcro:facebookbartbase-UseCase", hasTextValue: "You can use the raw model for text infilling. However, the model is mostly meant to be fine-tuned on a supervised dataset."});
MATCH (a {id: "mcro:mobilenetv3small100lambin1k"}), (b {id: "mcro:mobilenetv3small100lambin1k-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:mobilenetv3small100lambin1k-ModelDetail"}), (b {id: "mcro:mobilenetv3small100lambin1k-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:mobilenetv3small100lambin1k-ModelDetail"}), (b {id: "mcro:mobilenetv3small100lambin1k-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:mobilenetv3small100lambin1k-ModelDetail"}), (b {id: "mcro:mobilenetv3small100lambin1k-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:mobilenetv3small100lambin1k"}), (b {id: "mcro:mobilenetv3small100lambin1k-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:mobilenetv3small100lambin1k"}), (b {id: "mcro:mobilenetv3small100lambin1k-ModelParameter"}) CREATE (a)-[:HASMODELPARAMETER]->(b);
MATCH (a {id: "mcro:allMiniLML6v2"}), (b {id: "mcro:allMiniLML6v2-UseCaseInformationSection"}) CREATE (a)-[:HASUSECASEINFORMATIONSECTION]->(b);
MATCH (a {id: "mcro:allMiniLML6v2"}), (b {id: "mcro:allMiniLML6v2-TrainingDataInformationSection"}) CREATE (a)-[:HASTRAININGDATAINFORMATIONSECTION]->(b);
MATCH (a {id: "mcro:allMiniLML6v2"}), (b {id: "mcro:allMiniLML6v2-ModelArchitectureInformationSection"}) CREATE (a)-[:HASMODELARCHITECTUREINFORMATIONSECTION]->(b);
MATCH (a {id: "mcro:Falconsainsfwimagedetection"}), (b {id: "mcro:Falconsainsfwimagedetection-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:Falconsainsfwimagedetection"}), (b {id: "mcro:Falconsainsfwimagedetection-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:Falconsainsfwimagedetection"}), (b {id: "mcro:Falconsainsfwimagedetection-TrainingData"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:Falconsainsfwimagedetection"}), (b {id: "mcro:Falconsainsfwimagedetection-Reference"}) CREATE (a)-[:HASREFERENCE]->(b);
MATCH (a {id: "mcro:Falconsainsfwimagedetection-ModelDetail"}), (b {id: "mcro:Falconsainsfwimagedetection-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:Falconsainsfwimagedetection-ModelDetail"}), (b {id: "mcro:Falconsainsfwimagedetection-Architecture"}) CREATE (a)-[:HASARCHITECTURE]->(b);
MATCH (a {id: "mcro:Falconsainsfwimagedetection-UseCase"}), (b {id: "mcro:Falconsainsfwimagedetection-Limitation"}) CREATE (a)-[:HASLIMITATION]->(b);
MATCH (a {id: "mcro:dima806fairfaceageimagedetection"}), (b {id: "mcro:dima806fairfaceageimagedetection-Performance"}) CREATE (a)-[:HASPERFORMANCEMETRIC]->(b);
MATCH (a {id: "mcro:bertbasemodeluncased"}), (b {id: "mcro:bertbasemodeluncased-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:bertbasemodeluncased-ModelDetail"}), (b {id: "mcro:bertbasemodeluncased-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:bertbasemodeluncased-ModelDetail"}), (b {id: "mcro:bertbasemodeluncased-Architecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:bertbasemodeluncased"}), (b {id: "mcro:bertbasemodeluncased-IntendedUseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:bertbasemodeluncased"}), (b {id: "mcro:bertbasemodeluncased-Consideration"}) CREATE (a)-[:HASCONSIDERATION]->(b);
MATCH (a {id: "mcro:bertbasemodeluncased-Consideration"}), (b {id: "mcro:bertbasemodeluncased-Limitation"}) CREATE (a)-[:HASLIMITATION]->(b);
MATCH (a {id: "mcro:bertbasemodeluncased"}), (b {id: "mcro:bertbasemodeluncased-TrainingData"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:clip"}), (b {id: "mcro:clip-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:clip-ModelDetail"}), (b {id: "mcro:clip-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:clip-ModelDetail"}), (b {id: "mcro:clip-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:clip"}), (b {id: "mcro:clip-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:clip-UseCase"}), (b {id: "mcro:clip-PrimaryIntendedUseCase"}) CREATE (a)-[:HASPRIMARYINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:clip-UseCase"}), (b {id: "mcro:clip-OutOfScopeUseCase"}) CREATE (a)-[:HASOUTOFSCOPEUSECASE]->(b);
MATCH (a {id: "mcro:clip"}), (b {id: "mcro:clip-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:clip-Dataset"}), (b {id: "mcro:clip-TrainingData"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:clip"}), (b {id: "mcro:clip-QuantativeAnalysis"}) CREATE (a)-[:HASQUANTATIVEANALYSIS]->(b);
MATCH (a {id: "mcro:clip-QuantativeAnalysis"}), (b {id: "mcro:clip-Performance"}) CREATE (a)-[:HASPERFORMANCEMETRIC]->(b);
MATCH (a {id: "mcro:clip"}), (b {id: "mcro:clip-Consideration"}) CREATE (a)-[:HASCONSIDERATION]->(b);
MATCH (a {id: "mcro:clip-Consideration"}), (b {id: "mcro:clip-Limitation"}) CREATE (a)-[:HASLIMITATION]->(b);
MATCH (a {id: "mcro:clip-Consideration"}), (b {id: "mcro:clip-BiasAndFairness"}) CREATE (a)-[:HASETHICALCONSIDERATION]->(b);
MATCH (a {id: "mcro:TheBlokephi2GGUF"}), (b {id: "mcro:TheBlokephi2GGUF-Description"}) CREATE (a)-[:HASDESCRIPTION]->(b);
MATCH (a {id: "mcro:TheBlokephi2GGUF"}), (b {id: "mcro:TheBlokephi2GGUF-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:TheBlokephi2GGUF"}), (b {id: "mcro:TheBlokephi2GGUF-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:TheBlokephi2GGUF"}), (b {id: "mcro:TheBlokephi2GGUF-IntendedUseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:TheBlokephi2GGUF"}), (b {id: "mcro:TheBlokephi2GGUF-Limitation"}) CREATE (a)-[:HASLIMITATION]->(b);
MATCH (a {id: "mcro:TheBlokephi2GGUF"}), (b {id: "mcro:TheBlokephi2GGUF-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:TheBlokephi2GGUF"}), (b {id: "mcro:TheBlokephi2GGUF-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:chronos-t5-small"}), (b {id: "mcro:chronos-t5-small-architecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:chronos-t5-small"}), (b {id: "mcro:chronos-t5-small-citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:chronos-t5-small"}), (b {id: "mcro:chronos-t5-small-license"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:chronos-t5-small"}), (b {id: "mcro:chronos-t5-small-usecase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:chronos-t5-small"}), (b {id: "mcro:chronos-t5-small-dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:robertalargemodel"}), (b {id: "mcro:robertalargemodel-Architecture"}) CREATE (a)-[:HASARCHITECTURE]->(b);
MATCH (a {id: "mcro:robertalargemodel"}), (b {id: "mcro:robertalargemodel-UseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:robertalargemodel"}), (b {id: "mcro:robertalargemodel-TrainingData"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:ESMFold"}), (b {id: "mcro:ESMFold-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:ESMFold"}), (b {id: "mcro:ESMFold-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:YOLOv8DetectionModel"}), (b {id: "mcro:YOLOv8DetectionModel-DatasetInformationSection"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:YOLOv8DetectionModel"}), (b {id: "mcro:YOLOv8DetectionModel-UseCaseInformationSection"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:YOLOv8DetectionModel"}), (b {id: "mcro:YOLOv8DetectionModel-ModelArchitectureInformationSection"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:allmpnetbasev2"}), (b {id: "mcro:allmpnetbasev2-UseCaseInformationSection"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:allmpnetbasev2"}), (b {id: "mcro:allmpnetbasev2-TrainingDataInformationSection"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:allmpnetbasev2"}), (b {id: "mcro:allmpnetbasev2-ModelArchitectureInformationSection"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:electramodel"}), (b {id: "mcro:electramodel-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:electramodel"}), (b {id: "mcro:electramodel-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:electramodel"}), (b {id: "mcro:electramodel-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:pyannotewespeakervoxcelebresnet34LM"}), (b {id: "mcro:pyannotewespeakervoxcelebresnet34LM-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:pyannotewespeakervoxcelebresnet34LM"}), (b {id: "mcro:pyannotewespeakervoxcelebresnet34LM-Citation1"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:pyannotewespeakervoxcelebresnet34LM"}), (b {id: "mcro:pyannotewespeakervoxcelebresnet34LM-Citation2"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:pyannotewespeakervoxcelebresnet34LM"}), (b {id: "mcro:pyannotewespeakervoxcelebresnet34LM-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:resnet50a1in1k"}), (b {id: "mcro:resnet50a1in1k-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:resnet50a1in1k-ModelDetail"}), (b {id: "mcro:resnet50a1in1k-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:resnet50a1in1k-ModelDetail"}), (b {id: "mcro:resnet50a1in1k-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:resnet50a1in1k"}), (b {id: "mcro:resnet50a1in1k-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:resnet50a1in1k"}), (b {id: "mcro:resnet50a1in1k-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:llama318BInstructGGUF"}), (b {id: "mcro:llama318BInstructGGUF-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:llama318BInstructGGUF-ModelDetail"}), (b {id: "mcro:llama318BInstructGGUF-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:llama318BInstructGGUF-ModelDetail"}), (b {id: "mcro:llama318BInstructGGUF-Architecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:llama318BInstructGGUF"}), (b {id: "mcro:llama318BInstructGGUF-IntendedUse"}) CREATE (a)-[:HASINTENDEDUSE]->(b);
MATCH (a {id: "mcro:llama318BInstructGGUF"}), (b {id: "mcro:llama318BInstructGGUF-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:clip"}), (b {id: "mcro:clip-CitationInformationSection"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:clip"}), (b {id: "mcro:clip-ModelDetailSection"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:clip-ModelDetailSection"}), (b {id: "mcro:clip-VersionInformationSection"}) CREATE (a)-[:HASVERSION]->(b);
MATCH (a {id: "mcro:clip-ModelDetailSection"}), (b {id: "mcro:clip-ModelArchitectureInformationSection"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:clip-ModelDetailSection"}), (b {id: "mcro:clip-ReferenceInformationSection"}) CREATE (a)-[:HASREFERENCE]->(b);
MATCH (a {id: "mcro:clip"}), (b {id: "mcro:clip-UseCaseInformationSection"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:clip-UseCaseInformationSection"}), (b {id: "mcro:clip-UserInformationSection"}) CREATE (a)-[:HASUSER]->(b);
MATCH (a {id: "mcro:clip-UseCaseInformationSection"}), (b {id: "mcro:clip-ConsiderationInformationSection"}) CREATE (a)-[:HASCONSIDERATION]->(b);
MATCH (a {id: "mcro:clip"}), (b {id: "mcro:clip-DatasetInformationSection"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:clip-DatasetInformationSection"}), (b {id: "mcro:clip-ConsiderationInformationSection2"}) CREATE (a)-[:HASCONSIDERATION]->(b);
MATCH (a {id: "mcro:clip"}), (b {id: "mcro:clip-QuantativeAnalysisSection"}) CREATE (a)-[:HASQUANTATIVEANALYSIS]->(b);
MATCH (a {id: "mcro:clip-QuantativeAnalysisSection"}), (b {id: "mcro:clip-LimitationInformationSection"}) CREATE (a)-[:HASLIMITATION]->(b);
MATCH (a {id: "mcro:clip-QuantativeAnalysisSection"}), (b {id: "mcro:clip-RiskInformationSection"}) CREATE (a)-[:HASRISK]->(b);
MATCH (a {id: "mcro:pyannotesegmentation30"}), (b {id: "mcro:pyannotesegmentation30-UseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:pyannotesegmentation30"}), (b {id: "mcro:pyannotesegmentation30-InputFormat"}) CREATE (a)-[:HASINPUTFORMAT]->(b);
MATCH (a {id: "mcro:pyannotesegmentation30"}), (b {id: "mcro:pyannotesegmentation30-OutputFormat"}) CREATE (a)-[:HASOUTPUTFORMAT]->(b);
MATCH (a {id: "mcro:pyannotesegmentation30"}), (b {id: "mcro:pyannotesegmentation30-Dataset"}) CREATE (a)-[:HASTRAININGDATASET]->(b);
MATCH (a {id: "mcro:pyannotesegmentation30"}), (b {id: "mcro:pyannotesegmentation30-Citation1"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:pyannotesegmentation30"}), (b {id: "mcro:pyannotesegmentation30-Citation2"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:gpt2"}), (b {id: "mcro:gpt2-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:gpt2-ModelDetail"}), (b {id: "mcro:gpt2-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:gpt2"}), (b {id: "mcro:gpt2-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:gpt2"}), (b {id: "mcro:gpt2-Consideration"}) CREATE (a)-[:HASCONSIDERATION]->(b);
MATCH (a {id: "mcro:gpt2"}), (b {id: "mcro:gpt2-TrainingData"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:gpt2"}), (b {id: "mcro:gpt2-Evaluation"}) CREATE (a)-[:HASEVALUATION]->(b);
MATCH (a {id: "mcro:distilbertbasemodeluncased"}), (b {id: "mcro:distilbertbasemodeluncased-ModelDetailSection"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:distilbertbasemodeluncased-ModelDetailSection"}), (b {id: "mcro:distilbertbasemodeluncased-CitationInformationSection"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:distilbertbasemodeluncased-ModelDetailSection"}), (b {id: "mcro:distilbertbasemodeluncased-LicenseInformationSection"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:distilbertbasemodeluncased"}), (b {id: "mcro:distilbertbasemodeluncased-UseCaseInformationSection"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:distilbertbasemodeluncased"}), (b {id: "mcro:distilbertbasemodeluncased-TrainingDataInformationSection"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:distilbertbasemodeluncased"}), (b {id: "mcro:distilbertbasemodeluncased-LimitationInformationSection"}) CREATE (a)-[:HASLIMITATION]->(b);
MATCH (a {id: "mcro:clipsegModel"}), (b {id: "mcro:clipsegModel-UseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:clipsegModel"}), (b {id: "mcro:clipsegModel-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:pyannotespeakerdiarization31"}), (b {id: "mcro:pyannotespeakerdiarization31-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:pyannotespeakerdiarization31"}), (b {id: "mcro:pyannotespeakerdiarization31-Citation2"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:XLMROBERTaModel"}), (b {id: "mcro:XLMROBERTaModel-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:XLMROBERTaModel"}), (b {id: "mcro:XLMROBERTaModel-IntendedUseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:XLMROBERTaModel"}), (b {id: "mcro:XLMROBERTaModel-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:RoBERTa_base_model"}), (b {id: "mcro:RoBERTa_base_model-CitationInformationSection"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:RoBERTa_base_model"}), (b {id: "mcro:RoBERTa_base_model-ModelArchitectureInformationSection"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:RoBERTa_base_model"}), (b {id: "mcro:RoBERTa_base_model-UseCaseInformationSection"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:RoBERTa_base_model"}), (b {id: "mcro:RoBERTa_base_model-DatasetInformationSection"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:sentencetransformersparaphrasemultilingualMiniLML12v2"}), (b {id: "mcro:sentencetransformersparaphrasemultilingualMiniLML12v2-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:sentencetransformersparaphrasemultilingualMiniLML12v2"}), (b {id: "mcro:sentencetransformersparaphrasemultilingualMiniLML12v2-Architecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:sentencetransformersparaphrasemultilingualMiniLML12v2"}), (b {id: "mcro:sentencetransformersparaphrasemultilingualMiniLML12v2-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:chronosboltbase"}), (b {id: "mcro:chronosboltbase-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:chronosboltbase"}), (b {id: "mcro:chronosboltbase-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:chronosboltbase"}), (b {id: "mcro:chronosboltbase-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:chronosboltbase"}), (b {id: "mcro:chronosboltbase-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:sentencetransformersusecmlmmultilingual"}), (b {id: "mcro:sentencetransformersusecmlmmultilingual-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:sentencetransformersusecmlmmultilingual"}), (b {id: "mcro:sentencetransformersusecmlmmultilingual-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:whisperlargev3"}), (b {id: "mcro:whisperlargev3-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:whisperlargev3"}), (b {id: "mcro:whisperlargev3-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:whisperlargev3"}), (b {id: "mcro:whisperlargev3-TrainingData"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:whisperlargev3"}), (b {id: "mcro:whisperlargev3-IntendedUseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:whisperlargev3turbo"}), (b {id: "mcro:whisperlargev3turbo-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:whisperlargev3turbo"}), (b {id: "mcro:whisperlargev3turbo-Architecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:whisperlargev3turbo"}), (b {id: "mcro:whisperlargev3turbo-IntendedUseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:whisperlargev3turbo"}), (b {id: "mcro:whisperlargev3turbo-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:bertmultilingualbasemodelcased"}), (b {id: "mcro:bertmultilingualbasemodelcased-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:bertmultilingualbasemodelcased-ModelDetail"}), (b {id: "mcro:bertmultilingualbasemodelcased-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:bertmultilingualbasemodelcased"}), (b {id: "mcro:bertmultilingualbasemodelcased-UseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:bertmultilingualbasemodelcased"}), (b {id: "mcro:bertmultilingualbasemodelcased-TrainingData"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:bertmultilingualbasemodelcased"}), (b {id: "mcro:bertmultilingualbasemodelcased-TrainingProcedure"}) CREATE (a)-[:HASTRAININGPROCEDURE]->(b);
MATCH (a {id: "mcro:bertmultilingualbasemodelcased"}), (b {id: "mcro:bertmultilingualbasemodelcased-Citation2"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:vit-face-expression"}), (b {id: "mcro:vit-face-expression-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:vit-face-expression"}), (b {id: "mcro:vit-face-expression-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:vit-face-expression"}), (b {id: "mcro:vit-face-expression-Limitation"}) CREATE (a)-[:HASLIMITATION]->(b);
MATCH (a {id: "mcro:opt-125m"}), (b {id: "mcro:opt-125m-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:opt-125m-ModelDetail"}), (b {id: "mcro:opt-125m-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:opt-125m-ModelDetail"}), (b {id: "mcro:opt-125m-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:opt-125m-ModelDetail"}), (b {id: "mcro:opt-125m-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:opt-125m"}), (b {id: "mcro:opt-125m-IntendedUseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:opt-125m"}), (b {id: "mcro:opt-125m-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:opt-125m"}), (b {id: "mcro:opt-125m-Consideration"}) CREATE (a)-[:HASCONSIDERATION]->(b);
MATCH (a {id: "mcro:opt-125m"}), (b {id: "mcro:opt-125m-ModelParameter"}) CREATE (a)-[:HASMODELPARAMETER]->(b);
MATCH (a {id: "mcro:opt-125m"}), (b {id: "mcro:opt-125m-Training"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:siglipso400mpatch14384"}), (b {id: "mcro:siglipso400mpatch14384-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:siglipso400mpatch14384-ModelDetail"}), (b {id: "mcro:siglipso400mpatch14384-Architecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:siglipso400mpatch14384-ModelDetail"}), (b {id: "mcro:siglipso400mpatch14384-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:siglipso400mpatch14384"}), (b {id: "mcro:siglipso400mpatch14384-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:siglipso400mpatch14384"}), (b {id: "mcro:siglipso400mpatch14384-Parameter"}) CREATE (a)-[:HASMODELPARAMETER]->(b);
MATCH (a {id: "mcro:siglipso400mpatch14384-Parameter"}), (b {id: "mcro:siglipso400mpatch14384-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:chronosboltsmall"}), (b {id: "mcro:chronosboltsmall-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:chronosboltsmall"}), (b {id: "mcro:chronosboltsmall-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:chronosboltsmall"}), (b {id: "mcro:chronosboltsmall-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:chronosboltsmall"}), (b {id: "mcro:chronosboltsmall-IntendedUseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:metaLlama31"}), (b {id: "mcro:metaLlama31-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:metaLlama31-ModelDetail"}), (b {id: "mcro:metaLlama31-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:metaLlama31-ModelDetail"}), (b {id: "mcro:metaLlama31-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:metaLlama31"}), (b {id: "mcro:metaLlama31-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:metaLlama31-UseCase"}), (b {id: "mcro:metaLlama31-OutOfScopeUseCase"}) CREATE (a)-[:HASOUTOFSCOPEUSECASE]->(b);
MATCH (a {id: "mcro:metaLlama31"}), (b {id: "mcro:metaLlama31-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:distilbertbasemultilingualcased"}), (b {id: "mcro:distilbertbasemultilingualcased-ConsiderationInformationSection"}) CREATE (a)-[:HASCONSIDERATION]->(b);
MATCH (a {id: "mcro:distilbertbasemultilingualcased"}), (b {id: "mcro:distilbertbasemultilingualcased-CitationInformationSection"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:distilbertbasemultilingualcased"}), (b {id: "mcro:distilbertbasemultilingualcased-DatasetInformationSection"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:distilbertbasemultilingualcased"}), (b {id: "mcro:distilbertbasemultilingualcased-LicenseInformationSection"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:distilbertbasemultilingualcased"}), (b {id: "mcro:distilbertbasemultilingualcased-LimitationInformationSection"}) CREATE (a)-[:HASLIMITATION]->(b);
MATCH (a {id: "mcro:distilbertbasemultilingualcased"}), (b {id: "mcro:distilbertbasemultilingualcased-ModelArchitectureInformationSection"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:distilbertbasemultilingualcased"}), (b {id: "mcro:distilbertbasemultilingualcased-QuantativeAnalysisSection"}) CREATE (a)-[:HASQUANTATIVEANALYSIS]->(b);
MATCH (a {id: "mcro:distilbertbasemultilingualcased"}), (b {id: "mcro:distilbertbasemultilingualcased-UseCaseInformationSection"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:jonatasgrosmanwav2vec2largexlsr53portuguese"}), (b {id: "mcro:jonatasgrosmanwav2vec2largexlsr53portuguese-UseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:jonatasgrosmanwav2vec2largexlsr53portuguese"}), (b {id: "mcro:jonatasgrosmanwav2vec2largexlsr53portuguese-Architecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:jonatasgrosmanwav2vec2largexlsr53portuguese"}), (b {id: "mcro:jonatasgrosmanwav2vec2largexlsr53portuguese-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:jonatasgrosmanwav2vec2largexlsr53portuguese"}), (b {id: "mcro:jonatasgrosmanwav2vec2largexlsr53portuguese-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:XLMrobertalargeModel"}), (b {id: "mcro:XLMrobertalargeModel-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:XLMrobertalargeModel"}), (b {id: "mcro:XLMrobertalargeModel-Arch"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:XLMrobertalargeModel"}), (b {id: "mcro:XLMrobertalargeModel-UseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:clipvitlargepatch14336"}), (b {id: "mcro:clipvitlargepatch14336-ModelArchitectureInformationSection"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:clipvitlargepatch14336"}), (b {id: "mcro:clipvitlargepatch14336-DatasetInformationSection"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:clipvitlargepatch14336"}), (b {id: "mcro:clipvitlargepatch14336-UseCaseInformationSection"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:esm2"}), (b {id: "mcro:esm2-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:esm2"}), (b {id: "mcro:esm2-UseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:esm2"}), (b {id: "mcro:esm2-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:clip"}), (b {id: "mcro:clip-ModelDetailSection"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:clip-ModelDetailSection"}), (b {id: "mcro:clip-ModelArchitectureInformationSection"}) CREATE (a)-[:HASMODELARCHITECTUREINFORMATION]->(b);
MATCH (a {id: "mcro:clip-ModelDetailSection"}), (b {id: "mcro:clip-CitationInformationSection"}) CREATE (a)-[:HASCITATIONINFORMATION]->(b);
MATCH (a {id: "mcro:clip"}), (b {id: "mcro:clip-UseCaseInformationSection"}) CREATE (a)-[:HASUSECASEINFORMATION]->(b);
MATCH (a {id: "mcro:clip-UseCaseInformationSection"}), (b {id: "mcro:clip-PrimaryIntendedUseCaseInformationSection"}) CREATE (a)-[:HASPRIMARYINTENDEDUSECASEINFORMATION]->(b);
MATCH (a {id: "mcro:clip-UseCaseInformationSection"}), (b {id: "mcro:clip-OutofScopeUseCaseInformationSection"}) CREATE (a)-[:HASOUTOFSCOPEUSECASEINFORMATION]->(b);
MATCH (a {id: "mcro:clip"}), (b {id: "mcro:clip-DatasetInformationSection"}) CREATE (a)-[:HASDATASETINFORMATION]->(b);
MATCH (a {id: "mcro:clip"}), (b {id: "mcro:clip-ConsiderationInformationSection"}) CREATE (a)-[:HASCONSIDERATIONINFORMATION]->(b);
MATCH (a {id: "mcro:clip"}), (b {id: "mcro:clip-LimitationInformationSection"}) CREATE (a)-[:HASLIMITATIONINFORMATION]->(b);
MATCH (a {id: "mcro:jonatasgrosmanwav2vec2largexlsr53chinesezhcn"}), (b {id: "mcro:jonatasgrosmanwav2vec2largexlsr53chinesezhcn-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:jonatasgrosmanwav2vec2largexlsr53chinesezhcn"}), (b {id: "mcro:jonatasgrosmanwav2vec2largexlsr53chinesezhcn-Arch"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:jonatasgrosmanwav2vec2largexlsr53chinesezhcn"}), (b {id: "mcro:jonatasgrosmanwav2vec2largexlsr53chinesezhcn-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:jonatasgrosmanwav2vec2largexlsr53chinesezhcn"}), (b {id: "mcro:jonatasgrosmanwav2vec2largexlsr53chinesezhcn-UseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:t5base"}), (b {id: "mcro:t5base-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:t5base"}), (b {id: "mcro:t5base-ModelDetails"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:t5base"}), (b {id: "mcro:t5base-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:t5base"}), (b {id: "mcro:t5base-BiasRisksLimitations"}) CREATE (a)-[:HASCONSIDERATION]->(b);
MATCH (a {id: "mcro:t5base"}), (b {id: "mcro:t5base-TrainingDetails"}) CREATE (a)-[:HASMODELPARAMETER]->(b);
MATCH (a {id: "mcro:t5base"}), (b {id: "mcro:t5base-Evaluation"}) CREATE (a)-[:HASQUANTATIVEANALYSIS]->(b);
MATCH (a {id: "mcro:t5base"}), (b {id: "mcro:t5base-EnvironmentalImpact"}) CREATE (a)-[:HASCONSIDERATION]->(b);
MATCH (a {id: "mcro:t5base"}), (b {id: "mcro:t5base-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:t5base"}), (b {id: "mcro:t5base-ModelCardAuthors"}) CREATE (a)-[:HASOWNER]->(b);
MATCH (a {id: "mcro:distilbertbaseuncasedfinetunedsst2english"}), (b {id: "mcro:distilbertbaseuncasedfinetunedsst2english-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:distilbertbaseuncasedfinetunedsst2english"}), (b {id: "mcro:distilbertbaseuncasedfinetunedsst2english-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:distilbertbaseuncasedfinetunedsst2english"}), (b {id: "mcro:distilbertbaseuncasedfinetunedsst2english-Consideration"}) CREATE (a)-[:HASCONSIDERATION]->(b);
MATCH (a {id: "mcro:distilbertbaseuncasedfinetunedsst2english"}), (b {id: "mcro:distilbertbaseuncasedfinetunedsst2english-Training"}) CREATE (a)-[:HASMODELPARAMETER]->(b);
MATCH (a {id: "mcro:distilbertbaseuncasedfinetunedsst2english-ModelDetail"}), (b {id: "mcro:distilbertbaseuncasedfinetunedsst2english-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:multilinguale5small"}), (b {id: "mcro:multilinguale5small-CitationInformationSection"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:multilinguale5small"}), (b {id: "mcro:multilinguale5small-ModelArchitectureInformationSection"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:multilinguale5small"}), (b {id: "mcro:multilinguale5small-TrainingDataInformationSection"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:multilinguale5small"}), (b {id: "mcro:multilinguale5small-BenchmarkResults"}) CREATE (a)-[:HASBENCHMARK]->(b);
MATCH (a {id: "mcro:multilinguale5small"}), (b {id: "mcro:multilinguale5small-LimitationInformationSection"}) CREATE (a)-[:HASLIMITATION]->(b);
MATCH (a {id: "mcro:multilinguale5small"}), (b {id: "mcro:multilinguale5small-IntendedUseCaseInformationSection"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:visiontransformerbase"}), (b {id: "mcro:visiontransformerbase-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:visiontransformerbase-ModelDetail"}), (b {id: "mcro:visiontransformerbase-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:visiontransformerbase-ModelDetail"}), (b {id: "mcro:visiontransformerbase-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:visiontransformerbase"}), (b {id: "mcro:visiontransformerbase-ModelParameter"}) CREATE (a)-[:HASMODELPARAMETER]->(b);
MATCH (a {id: "mcro:visiontransformerbase-ModelParameter"}), (b {id: "mcro:visiontransformerbase-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:visiontransformerbase"}), (b {id: "mcro:visiontransformerbase-IntendedUseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:visiontransformerbase"}), (b {id: "mcro:visiontransformerbase-TrainingData"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:visiontransformerbase"}), (b {id: "mcro:visiontransformerbase-TrainingProcedure"}) CREATE (a)-[:HASTRAININGPROCEDURE]->(b);
MATCH (a {id: "mcro:visiontransformerbase"}), (b {id: "mcro:visiontransformerbase-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:bertbasemodelcased"}), (b {id: "mcro:bertbasemodelcased-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:bertbasemodelcased-ModelDetail"}), (b {id: "mcro:bertbasemodelcased-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:bertbasemodelcased-ModelDetail"}), (b {id: "mcro:bertbasemodelcased-Architecture"}) CREATE (a)-[:HASARCHITECTURE]->(b);
MATCH (a {id: "mcro:bertbasemodelcased-ModelDetail"}), (b {id: "mcro:bertbasemodelcased-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:bertbasemodelcased"}), (b {id: "mcro:bertbasemodelcased-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:bertbasemodelcased"}), (b {id: "mcro:bertbasemodelcased-Consideration"}) CREATE (a)-[:HASCONSIDERATION]->(b);
MATCH (a {id: "mcro:bertbasemodelcased"}), (b {id: "mcro:bertbasemodelcased-TrainingData"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:bertbasemodelcased"}), (b {id: "mcro:bertbasemodelcased-Parameter"}) CREATE (a)-[:HASPARAMETER]->(b);
MATCH (a {id: "mcro:bertbasemodelcased"}), (b {id: "mcro:bertbasemodelcased-Evaluation"}) CREATE (a)-[:HASEVALUATION]->(b);
MATCH (a {id: "mcro:bertbasemodelcased-Evaluation"}), (b {id: "mcro:bertbasemodelcased-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:jinaaijinaembeddingsv3"}), (b {id: "mcro:jinaaijinaembeddingsv3-UseCaseInformationSection"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:jinaaijinaembeddingsv3"}), (b {id: "mcro:jinaaijinaembeddingsv3-ModelArchitectureInformationSection"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:jinaaijinaembeddingsv3"}), (b {id: "mcro:jinaaijinaembeddingsv3-LicenseInformationSection"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:jinaaijinaembeddingsv3"}), (b {id: "mcro:jinaaijinaembeddingsv3-CitationInformationSection"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:sentencetransformersparaphraseMiniLML6v2"}), (b {id: "mcro:sentencetransformersparaphraseMiniLML6v2-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:sentencetransformersparaphraseMiniLML6v2"}), (b {id: "mcro:sentencetransformersparaphraseMiniLML6v2-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:resnet18a1in1k"}), (b {id: "mcro:resnet18a1in1k-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:resnet18a1in1k-ModelDetail"}), (b {id: "mcro:resnet18a1in1k-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:resnet18a1in1k"}), (b {id: "mcro:resnet18a1in1k-Architecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:resnet18a1in1k"}), (b {id: "mcro:resnet18a1in1k-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:resnet18a1in1k"}), (b {id: "mcro:resnet18a1in1k-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:flant5base"}), (b {id: "mcro:flant5base-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:flant5base-ModelDetail"}), (b {id: "mcro:flant5base-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:flant5base-ModelDetail"}), (b {id: "mcro:flant5base-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:flant5base"}), (b {id: "mcro:flant5base-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:flant5base"}), (b {id: "mcro:flant5base-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:fashionclip"}), (b {id: "mcro:fashionclip-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:fashionclip-ModelDetail"}), (b {id: "mcro:fashionclip-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:fashionclip-ModelDetail"}), (b {id: "mcro:fashionclip-Architecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:fashionclip"}), (b {id: "mcro:fashionclip-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:fashionclip"}), (b {id: "mcro:fashionclip-Consideration"}) CREATE (a)-[:HASCONSIDERATION]->(b);
MATCH (a {id: "mcro:jonatasgrosmanwav2vec2largexlsr53russian"}), (b {id: "mcro:jonatasgrosmanwav2vec2largexlsr53russian-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:jonatasgrosmanwav2vec2largexlsr53russian"}), (b {id: "mcro:jonatasgrosmanwav2vec2largexlsr53russian-Architecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:jonatasgrosmanwav2vec2largexlsr53russian"}), (b {id: "mcro:jonatasgrosmanwav2vec2largexlsr53russian-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:jonatasgrosmanwav2vec2largexlsr53russian"}), (b {id: "mcro:jonatasgrosmanwav2vec2largexlsr53russian-UseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:twitterrobertabasesentiment"}), (b {id: "mcro:twitterrobertabasesentiment-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:twitterrobertabasesentiment"}), (b {id: "mcro:twitterrobertabasesentiment-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:twitterrobertabasesentiment"}), (b {id: "mcro:twitterrobertabasesentiment-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:twitterrobertabasesentiment"}), (b {id: "mcro:twitterrobertabasesentiment-IntendedUseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:vitmatte"}), (b {id: "mcro:vitmatte-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:vitmatte-ModelDetail"}), (b {id: "mcro:vitmatte-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:vitmatte-ModelDetail"}), (b {id: "mcro:vitmatte-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:vitmatte"}), (b {id: "mcro:vitmatte-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:FlagEmbedding"}), (b {id: "mcro:FlagEmbedding-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:FlagEmbedding"}), (b {id: "mcro:FlagEmbedding-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:bartlargecnn"}), (b {id: "mcro:bartlargecnn-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:bartlargecnn-ModelDetail"}), (b {id: "mcro:bartlargecnn-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:bartlargecnn-ModelDetail"}), (b {id: "mcro:bartlargecnn-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:bartlargecnn"}), (b {id: "mcro:bartlargecnn-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:bartlargecnn"}), (b {id: "mcro:bartlargecnn-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:stablediffusionv15modelcard"}), (b {id: "mcro:stablediffusionv15modelcard-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:stablediffusionv15modelcard"}), (b {id: "mcro:stablediffusionv15modelcard-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:stablediffusionv15modelcard"}), (b {id: "mcro:stablediffusionv15modelcard-IntendedUseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:stablediffusionv15modelcard"}), (b {id: "mcro:stablediffusionv15modelcard-Architecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:stablediffusionv15modelcard"}), (b {id: "mcro:stablediffusionv15modelcard-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:BGE-M3"}), (b {id: "mcro:BGE-M3-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:BGE-M3"}), (b {id: "mcro:BGE-M3-Architecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:BGE-M3"}), (b {id: "mcro:BGE-M3-UseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:BGE-M3"}), (b {id: "mcro:BGE-M3-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:YOLOWorldMirror"}), (b {id: "mcro:YOLOWorldMirror-Documentation"}) CREATE (a)-[:HASDOCUMENTATION]->(b);
MATCH (a {id: "mcro:bertbasechinese"}), (b {id: "mcro:bertbasechinese-ModelDetailSection"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:bertbasechinese-ModelDetailSection"}), (b {id: "mcro:bertbasechinese-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:bertbasechinese"}), (b {id: "mcro:bertbasechinese-UseCaseSection"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:bertbasechinese"}), (b {id: "mcro:bertbasechinese-ConsiderationSection"}) CREATE (a)-[:HASCONSIDERATION]->(b);
MATCH (a {id: "mcro:bertbasechinese"}), (b {id: "mcro:bertbasechinese-ModelParameterSection"}) CREATE (a)-[:HASMODELPARAMETER]->(b);
MATCH (a {id: "mcro:bertbasechinese"}), (b {id: "mcro:bertbasechinese-QuantativeAnalysisSection"}) CREATE (a)-[:HASQUANTATIVEANALYSIS]->(b);
MATCH (a {id: "mcro:bartlargemnli"}), (b {id: "mcro:bartlargemnli-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:bartlargemnli"}), (b {id: "mcro:facebookbartlargemnli-Citation2"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:bartlargemnli"}), (b {id: "mcro:MultiNLI"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:bartlargemnli"}), (b {id: "mcro:BART"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:bartlargemnli"}), (b {id: "mcro:NLIbasedZeroShotTextClassification"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:CLIPViTB16LAION2B"}), (b {id: "mcro:CLIPViTB16LAION2B-DatasetInfo"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:CLIPViTB16LAION2B"}), (b {id: "mcro:CLIPViTB16LAION2B-ModelArch"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:CLIPViTB16LAION2B"}), (b {id: "mcro:CLIPViTB16LAION2B-LicenseInfo"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:CLIPViTB16LAION2B"}), (b {id: "mcro:CLIPViTB16LAION2B-CitationInfo"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:CLIPViTB16LAION2B"}), (b {id: "mcro:CLIPViTB16LAION2B-UseCaseInfo"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:stablediffusioninpaintingmodelcard"}), (b {id: "mcro:stablediffusioninpaintingmodelcard-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:stablediffusioninpaintingmodelcard"}), (b {id: "mcro:stablediffusioninpaintingmodelcard-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:stablediffusioninpaintingmodelcard"}), (b {id: "mcro:stablediffusioninpaintingmodelcard-Limitation"}) CREATE (a)-[:HASLIMITATION]->(b);
MATCH (a {id: "mcro:stablediffusioninpaintingmodelcard"}), (b {id: "mcro:stablediffusioninpaintingmodelcard-TrainingData"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:stablediffusioninpaintingmodelcard"}), (b {id: "mcro:stablediffusioninpaintingmodelcard-EvaluationResult"}) CREATE (a)-[:HASEVALUATIONRESULT]->(b);
MATCH (a {id: "mcro:stablediffusioninpaintingmodelcard"}), (b {id: "mcro:stablediffusioninpaintingmodelcard-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:stablediffusioninpaintingmodelcard-ModelDetail"}), (b {id: "mcro:stablediffusioninpaintingmodelcard-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:stablediffusioninpaintingmodelcard-ModelDetail"}), (b {id: "mcro:stablediffusioninpaintingmodelcard-Architecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:stablediffusioninpaintingmodelcard-TrainingData"}), (b {id: "mcro:stablediffusioninpaintingmodelcard-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:Qwen2505BInstruct"}), (b {id: "mcro:Qwen2505BInstruct-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:Qwen2505BInstruct"}), (b {id: "mcro:Qwen2505BInstruct-Architecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:Qwen2505BInstruct"}), (b {id: "mcro:Qwen2505BInstruct-Parameter"}) CREATE (a)-[:HASMODELPARAMETER]->(b);
MATCH (a {id: "mcro:Qwen257BInstruct"}), (b {id: "mcro:Qwen257BInstruct-Architecture"}) CREATE (a)-[:HASARCHITECTURE]->(b);
MATCH (a {id: "mcro:Qwen257BInstruct"}), (b {id: "mcro:Qwen257BInstruct-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:clipViTbigG14laion2B"}), (b {id: "mcro:clipViTbigG14laion2B-ModelDetailSection"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:clipViTbigG14laion2B-ModelDetailSection"}), (b {id: "mcro:clipViTbigG14laion2B-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:clipViTbigG14laion2B-ModelDetailSection"}), (b {id: "mcro:clipViTbigG14laion2B-CitationLAION"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:clipViTbigG14laion2B-ModelDetailSection"}), (b {id: "mcro:clipViTbigG14laion2B-CitationOpenAICLIP"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:clipViTbigG14laion2B-ModelDetailSection"}), (b {id: "mcro:clipViTbigG14laion2B-CitationOpenCLIP"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:clipViTbigG14laion2B-ModelDetailSection"}), (b {id: "mcro:clipViTbigG14laion2B-CitationScalingOpenCLIP"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:clipViTbigG14laion2B"}), (b {id: "mcro:clipViTbigG14laion2B-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:clipViTbigG14laion2B"}), (b {id: "mcro:clipViTbigG14laion2B-ModelParameter"}) CREATE (a)-[:HASMODELPARAMETER]->(b);
MATCH (a {id: "mcro:clipViTbigG14laion2B-ModelParameter"}), (b {id: "mcro:clipViTbigG14laion2B-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:clipViTbigG14laion2B"}), (b {id: "mcro:clipViTbigG14laion2B-QuantativeAnalysis"}) CREATE (a)-[:HASQUANTATIVEANALYSIS]->(b);
MATCH (a {id: "mcro:allMiniLML12v2"}), (b {id: "mcro:allMiniLML12v2-UseCaseInformationSection"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:allMiniLML12v2"}), (b {id: "mcro:allMiniLML12v2-TrainingDataInformationSection"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:allMiniLML12v2"}), (b {id: "mcro:allMiniLML12v2-ModelArchitectureInformationSection"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:tabletransformer"}), (b {id: "mcro:tabletransformer-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:tabletransformer-ModelDetail"}), (b {id: "mcro:tabletransformer-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:tabletransformer-ModelDetail"}), (b {id: "mcro:tabletransformer-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:tabletransformer"}), (b {id: "mcro:tabletransformer-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:t5small"}), (b {id: "mcro:t5small-ModelDetails"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:t5small-ModelDetails"}), (b {id: "mcro:t5small-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:t5small-ModelDetails"}), (b {id: "mcro:t5small-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:t5small"}), (b {id: "mcro:t5small-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:t5small"}), (b {id: "mcro:t5small-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:t5small-ModelDetails"}), (b {id: "mcro:t5small-Architecture"}) CREATE (a)-[:HASARCHITECTURE]->(b);
MATCH (a {id: "mcro:albertbasev2"}), (b {id: "mcro:albertbasev2-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:albertbasev2-ModelDetail"}), (b {id: "mcro:albertbasev2-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:albertbasev2-ModelDetail"}), (b {id: "mcro:albertbasev2-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:albertbasev2-ModelDetail"}), (b {id: "mcro:albertbasev2-Architecture"}) CREATE (a)-[:HASARCHITECTURE]->(b);
MATCH (a {id: "mcro:albertbasev2"}), (b {id: "mcro:albertbasev2-UseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:albertbasev2"}), (b {id: "mcro:albertbasev2-TrainingData"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:albertbasev2"}), (b {id: "mcro:albertbasev2-Limitation"}) CREATE (a)-[:HASLIMITATION]->(b);
MATCH (a {id: "mcro:distilgpt2"}), (b {id: "mcro:distilgpt2-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:distilgpt2-ModelDetail"}), (b {id: "mcro:distilgpt2-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:distilgpt2-ModelDetail"}), (b {id: "mcro:distilgpt2-Architecture"}) CREATE (a)-[:HASARCHITECTURE]->(b);
MATCH (a {id: "mcro:distilgpt2"}), (b {id: "mcro:distilgpt2-IntendedUse"}) CREATE (a)-[:HASINTENDEDUSE]->(b);
MATCH (a {id: "mcro:distilgpt2"}), (b {id: "mcro:distilgpt2-Limitation"}) CREATE (a)-[:HASLIMITATION]->(b);
MATCH (a {id: "mcro:distilgpt2"}), (b {id: "mcro:distilgpt2-TrainingData"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:distilgpt2"}), (b {id: "mcro:distilgpt2-TrainingProcedure"}) CREATE (a)-[:HASTRAININGPROCEDURE]->(b);
MATCH (a {id: "mcro:distilgpt2"}), (b {id: "mcro:distilgpt2-EvaluationResult"}) CREATE (a)-[:HASEVALUATIONRESULT]->(b);
MATCH (a {id: "mcro:distilgpt2"}), (b {id: "mcro:distilgpt2-EnvironmentalImpact"}) CREATE (a)-[:HASENVIRONMENTALIMPACT]->(b);
MATCH (a {id: "mcro:distilgpt2"}), (b {id: "mcro:distilgpt2-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:forcedalignerwithhuggingfacectcmodels"}), (b {id: "mcro:forcedalignerwithhuggingfacectcmodels-intendedusecase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:forcedalignerwithhuggingfacectcmodels"}), (b {id: "mcro:forcedalignerwithhuggingfacectcmodels-modelarchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:mixedbreadaimxbaiembedlargev1"}), (b {id: "mcro:mixedbreadaimxbaiembedlargev1-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:mixedbreadaimxbaiembedlargev1"}), (b {id: "mcro:mixedbreadaimxbaiembedlargev1-Citation1"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:mixedbreadaimxbaiembedlargev1"}), (b {id: "mcro:mixedbreadaimxbaiembedlargev1-Citation2"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:mixedbreadaimxbaiembedlargev1"}), (b {id: "mcro:mixedbreadaimxbaiembedlargev1-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:mixedbreadaimxbaiembedlargev1"}), (b {id: "mcro:mixedbreadaimxbaiembedlargev1-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:mixedbreadaimxbaiembedlargev1"}), (b {id: "mcro:mixedbreadaimxbaiembedlargev1-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:facebookdinov2small"}), (b {id: "mcro:facebookdinov2small-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:facebookdinov2small-ModelDetail"}), (b {id: "mcro:facebookdinov2small-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:facebookdinov2small-ModelDetail"}), (b {id: "mcro:facebookdinov2small-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:facebookdinov2small-ModelDetail"}), (b {id: "mcro:facebookdinov2small-Architecture"}) CREATE (a)-[:HASARCHITECTURE]->(b);
MATCH (a {id: "mcro:facebookdinov2small"}), (b {id: "mcro:facebookdinov2small-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:efficientnet_b3ra2in1k"}), (b {id: "mcro:efficientnet_b3ra2in1k-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:efficientnet_b3ra2in1k"}), (b {id: "mcro:efficientnet_b3ra2in1k-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:efficientnet_b3ra2in1k"}), (b {id: "mcro:efficientnet_b3ra2in1k-Citation-1"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:efficientnet_b3ra2in1k"}), (b {id: "mcro:efficientnet_b3ra2in1k-Citation-2"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:efficientnet_b3ra2in1k"}), (b {id: "mcro:efficientnet_b3ra2in1k-Citation-3"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:efficientnet_b3ra2in1k"}), (b {id: "mcro:efficientnet_b3ra2in1k-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:Qwen25VL7BInstruct"}), (b {id: "mcro:Qwen25VL7BInstruct-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:Qwen25VL7BInstruct"}), (b {id: "mcro:Qwen25VL7BInstruct-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:Qwen25VL7BInstruct"}), (b {id: "mcro:Qwen25VL7BInstruct-IntendedUseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:Qwen25VL7BInstruct"}), (b {id: "mcro:Qwen25VL7BInstruct-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:gemma3"}), (b {id: "mcro:gemma3-ModelDetailSection"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:gemma3-ModelDetailSection"}), (b {id: "mcro:gemma3-CitationSection"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:gemma3"}), (b {id: "mcro:gemma3-ModelParameterSection"}) CREATE (a)-[:HASMODELPARAMETER]->(b);
MATCH (a {id: "mcro:gemma3-ModelParameterSection"}), (b {id: "mcro:gemma3-DatasetSection"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:gemma3-ModelParameterSection"}), (b {id: "mcro:gemma3-ModelArchitectureSection"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:gemma3"}), (b {id: "mcro:gemma3-QuantativeAnalysisSection"}) CREATE (a)-[:HASQUANTATIVEANALYSIS]->(b);
MATCH (a {id: "mcro:gemma3"}), (b {id: "mcro:gemma3-ConsiderationSection"}) CREATE (a)-[:HASCONSIDERATION]->(b);
MATCH (a {id: "mcro:gemma3"}), (b {id: "mcro:gemma3-UseCaseSection"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:gemma3"}), (b {id: "mcro:gemma3-ModelCardReport"}) CREATE (a)-[:HASMODELCARDREPORT]->(b);
MATCH (a {id: "mcro:sentencetransformersparaphrasemultilingualmpnetbasv2"}), (b {id: "mcro:sentencetransformersparaphrasemultilingualmpnetbasv2-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:sentencetransformersparaphrasemultilingualmpnetbasv2"}), (b {id: "mcro:sentencetransformersparaphrasemultilingualmpnetbasv2-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:sentencetransformersparaphrasemultilingualmpnetbasv2"}), (b {id: "mcro:sentencetransformersparaphrasemultilingualmpnetbasv2-UseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:Salesforceblipbootstrapping"}), (b {id: "mcro:Salesforceblipbootstrapping-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:Salesforceblipbootstrapping"}), (b {id: "mcro:Salesforceblipbootstrapping-UseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:Salesforceblipbootstrapping"}), (b {id: "mcro:Salesforceblipbootstrapping-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:Salesforceblipbootstrapping"}), (b {id: "mcro:Salesforceblipbootstrapping-Architecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:Salesforceblipbootstrapping"}), (b {id: "mcro:Salesforceblipbootstrapping-Consideration"}) CREATE (a)-[:HASCONSIDERATION]->(b);
MATCH (a {id: "mcro:Salesforceblipbootstrapping"}), (b {id: "mcro:cocoDataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:flux1dev"}), (b {id: "mcro:flux1dev-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:flux1dev"}), (b {id: "mcro:flux1dev-Limitation"}) CREATE (a)-[:HASLIMITATION]->(b);
MATCH (a {id: "mcro:flux1dev"}), (b {id: "mcro:flux1dev-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:flux1dev"}), (b {id: "mcro:flux1dev-OutOfScopeUse"}) CREATE (a)-[:HASOUTOFSCOPEUSE]->(b);
MATCH (a {id: "mcro:flux1dev"}), (b {id: "mcro:flux1dev-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:unik3d"}), (b {id: "mcro:unik3d-Library"}) CREATE (a)-[:HASLIBRARY]->(b);
MATCH (a {id: "mcro:unik3d"}), (b {id: "mcro:unik3d-Documentation"}) CREATE (a)-[:HASDOCUMENTATION]->(b);
MATCH (a {id: "mcro:llama32"}), (b {id: "mcro:llama32-modelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:llama32-modelDetail"}), (b {id: "mcro:llama32-license"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:llama32-modelDetail"}), (b {id: "mcro:llama32-citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:llama32-modelDetail"}), (b {id: "mcro:llama32-architecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:llama32"}), (b {id: "mcro:llama32-usecase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:llama32-usecase"}), (b {id: "mcro:llama32-outofscope"}) CREATE (a)-[:HASOUTOFSCOPEUSECASE]->(b);
MATCH (a {id: "mcro:llama32"}), (b {id: "mcro:llama32-parameter"}) CREATE (a)-[:HASMODELPARAMETER]->(b);
MATCH (a {id: "mcro:llama32-parameter"}), (b {id: "mcro:llama32-trainingdata"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:bertmultilingualbasemodeluncased"}), (b {id: "mcro:bertmultilingualbasemodeluncased-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:bertmultilingualbasemodeluncased"}), (b {id: "mcro:bertmultilingualbasemodeluncased-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:bertmultilingualbasemodeluncased"}), (b {id: "mcro:bertmultilingualbasemodeluncased-IntendedUseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:bertmultilingualbasemodeluncased"}), (b {id: "mcro:bertmultilingualbasemodeluncased-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:Salesforceblipimagecaptioninglarge"}), (b {id: "mcro:Salesforceblipimagecaptioninglarge-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:Salesforceblipimagecaptioninglarge"}), (b {id: "mcro:Salesforceblipimagecaptioninglarge-IntendedUseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:Salesforceblipimagecaptioninglarge"}), (b {id: "mcro:Salesforceblipimagecaptioninglarge-Considerations"}) CREATE (a)-[:HASCONSIDERATIONS]->(b);
MATCH (a {id: "mcro:Salesforceblipimagecaptioninglarge"}), (b {id: "mcro:Salesforceblipimagecaptioninglarge-Architecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:Salesforceblipimagecaptioninglarge"}), (b {id: "mcro:Salesforceblipimagecaptioninglarge-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:crossencodermsmarcoMiniLML6v2"}), (b {id: "mcro:crossencodermsmarcoMiniLML6v2-UseCase"}) CREATE (a)-[:HASUSECASEINFORMATIONSECTION]->(b);
MATCH (a {id: "mcro:crossencodermsmarcoMiniLML6v2"}), (b {id: "mcro:crossencodermsmarcoMiniLML6v2-Performance"}) CREATE (a)-[:HASPERFORMANCEMETRICINFORMATIONSECTION]->(b);
MATCH (a {id: "mcro:sdxl10base"}), (b {id: "mcro:sdxl10base-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:sdxl10base-ModelDetail"}), (b {id: "mcro:sdxl10base-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:sdxl10base-ModelDetail"}), (b {id: "mcro:sdxl10base-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:sdxl10base-ModelDetail"}), (b {id: "mcro:sdxl10base-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:sdxl10base"}), (b {id: "mcro:sdxl10base-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:sdxl10base"}), (b {id: "mcro:sdxl10base-Consideration"}) CREATE (a)-[:HASCONSIDERATION]->(b);
MATCH (a {id: "mcro:sdxl10base-Consideration"}), (b {id: "mcro:sdxl10base-Limitation"}) CREATE (a)-[:HASLIMITATION]->(b);
MATCH (a {id: "mcro:sdxl10base-Consideration"}), (b {id: "mcro:sdxl10base-Risk"}) CREATE (a)-[:HASRISK]->(b);
MATCH (a {id: "mcro:sdxl10base"}), (b {id: "mcro:sdxl10base-OutOfScopeUseCase"}) CREATE (a)-[:HASOUTOFSCOPEUSECASE]->(b);
MATCH (a {id: "mcro:sdxl10base"}), (b {id: "mcro:sdxl10base-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:whisper"}), (b {id: "mcro:whisper-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:whisper-ModelDetail"}), (b {id: "mcro:whisper-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:whisper-ModelDetail"}), (b {id: "mcro:whisper-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:whisper"}), (b {id: "mcro:whisper-IntendedUseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:whisper"}), (b {id: "mcro:whisper-TrainingData"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:jonatasgrosmanwav2vec2largexlsr53dutch"}), (b {id: "mcro:jonatasgrosmanwav2vec2largexlsr53dutch-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:jonatasgrosmanwav2vec2largexlsr53dutch"}), (b {id: "mcro:jonatasgrosmanwav2vec2largexlsr53dutch-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:jonatasgrosmanwav2vec2largexlsr53dutch"}), (b {id: "mcro:jonatasgrosmanwav2vec2largexlsr53dutch-Dataset"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:jonatasgrosmanwav2vec2largexlsr53dutch"}), (b {id: "mcro:jonatasgrosmanwav2vec2largexlsr53dutch-Dataset2"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:jonatasgrosmanwav2vec2largexlsr53dutch"}), (b {id: "mcro:jonatasgrosmanwav2vec2largexlsr53dutch-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:visiontransformerbase"}), (b {id: "mcro:visiontransformerbase-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:visiontransformerbase"}), (b {id: "mcro:visiontransformerbase-Citation1"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:visiontransformerbase"}), (b {id: "mcro:visiontransformerbase-IntendedUseCase"}) CREATE (a)-[:HASINTENDEDUSECASE]->(b);
MATCH (a {id: "mcro:visiontransformerbase"}), (b {id: "mcro:visiontransformerbase-TrainingData"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:visiontransformerbase"}), (b {id: "mcro:visiontransformerbase-Citation2"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:visiontransformerbase"}), (b {id: "mcro:visiontransformerbase-Citation3"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:clinicalBERT-Bio+ClinicalBERTModel"}), (b {id: "mcro:clinicalBERT-Bio+ClinicalBERTModel-DatasetInfo"}) CREATE (a)-[:HASDATASET]->(b);
MATCH (a {id: "mcro:clinicalBERT-Bio+ClinicalBERTModel"}), (b {id: "mcro:clinicalBERT-Bio+ClinicalBERTModel-ArchInfo"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:clinicalBERT-Bio+ClinicalBERTModel"}), (b {id: "mcro:clinicalBERT-Bio+ClinicalBERTModel-CitationInfo"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:clinicalBERT-Bio+ClinicalBERTModel"}), (b {id: "mcro:clinicalBERT-Bio+ClinicalBERTModel-UseCaseInfo"}) CREATE (a)-[:HASUSECASE]->(b);
MATCH (a {id: "mcro:clinicalBERT-Bio+ClinicalBERTModel"}), (b {id: "mcro:clinicalBERT-Bio+ClinicalBERTModel-TrainingDataInfo"}) CREATE (a)-[:HASTRAININGDATA]->(b);
MATCH (a {id: "mcro:FlagEmbedding"}), (b {id: "mcro:FlagEmbedding-CitationInformationSection"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:FlagEmbedding"}), (b {id: "mcro:FlagEmbedding-LicenseInformationSection"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:FlagEmbedding"}), (b {id: "mcro:FlagEmbedding-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:FlagEmbedding"}), (b {id: "mcro:FlagEmbedding-License"}) CREATE (a)-[:HASLICENSE]->(b);
MATCH (a {id: "mcro:facebookbartbase"}), (b {id: "mcro:facebookbartbase-ModelDetail"}) CREATE (a)-[:HASMODELDETAIL]->(b);
MATCH (a {id: "mcro:facebookbartbase-ModelDetail"}), (b {id: "mcro:facebookbartbase-Citation"}) CREATE (a)-[:HASCITATION]->(b);
MATCH (a {id: "mcro:facebookbartbase-ModelDetail"}), (b {id: "mcro:facebookbartbase-ModelArchitecture"}) CREATE (a)-[:HASMODELARCHITECTURE]->(b);
MATCH (a {id: "mcro:facebookbartbase"}), (b {id: "mcro:facebookbartbase-UseCase"}) CREATE (a)-[:HASUSECASE]->(b);