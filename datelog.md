# Work Log for AI Walkability Assessment (MOST)

The goal of the project is to train AI models for the assessment of Walkability Labels. This project started in February 2024, and definetely needs some mantainance.
This particular repo is related to the AI backbone and the only focus is model training and result logging.
The novelty idea of this research is to use Satellite pictures for Walkability assessment joint to traditional Streetview pictures. 
Originally the idea was to use two transformers, one for each type of data source and then have a CNN give an evaluation based on what each model has learned.
The idea has evolved into using a Dual Encoder approach which embeds everything into a single model, each model is actually an image encoder which is trained to extract relevant features.
Recently, I tried submitting an article into IEEE Explore, and received desk reject with technical reviews. So, right now is the time to get hold of this code and optimize.
Moreover, this repo has accumulated useless files.

## 26/05/25
The goal of this day is to remove all old files (excels for results, example photos) and keep just the relevant facts to then update the repo. Moreover, I will read the review to figure out viable plans to address the critiques received, realted to the AI backbone. 
Now that the repository is set, I have to guarantee myself I would be able to work from home, since tomorrow there is no power in the university building. So, I have to make sure I can clone the repo and work perfectly from there.
After doing that, I'll use the trained model to perform inference with the area proposed by Palermo University.

### Tasks Done
- Deleting unnecessary files from the folder
- Updating git repo
- Adding inference functions to inference.py

### Decisions Made
- [Decision, reason, alternatives considered]

### Problems & Solutions
- [Any issues and how you solved them]

### Thoughts/Next Steps
- Update the readme.md in github with a newer description
- Address imbalance of the dataset using class-weighted loss-function/oversampling/specialized regression techniques
- Benchmarking only against transformer variants and not using standard CNN-based models. (High)
- Testing or cross-city validation to demonstrate the model generalizes to other urban contexts (Low)


