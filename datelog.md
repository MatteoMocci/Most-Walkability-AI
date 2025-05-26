# Work Log for AI Walkability Assessment (MOST)

The goal of the project is to train AI models for the assessment of Walkability Labels. This project started in February 2024, and definetely needs some mantainance.
This particular repo is related to the AI backbone and the only focus is model training and result logging.
The novelty idea of this research is to use Satellite pictures for Walkability assessment joint to traditional Streetview pictures. 
Originally the idea was to use two transformers, one for each type of data source and then have a CNN give an evaluation based on what each model has learned.
The idea has evolved into using a Dual Encoder approach which embeds everything into a single model, each model is actually an image encoder which is trained to extract relevant features.
Recently, I tried submitting an article into IEEE Explore, and received desk reject with technical reviews. So, right now is the time to get hold of this code and optimize.
Moreover, this repo has accumulated useless files.

## 26/05/25
The goal of this day is to remove all old files (excels for results, example photos) and keep just the relevant facts to then update the repo. Moreover, I will read the critiques to figure out viable plans to address the critiques received.
### Tasks Done
- Deleting unnecessary files from the folder
- Updating git repo

### Decisions Made
- [Decision, reason, alternatives considered]

### Problems & Solutions
- [Any issues and how you solved them]

### Thoughts/Next Steps
- [Ideas for improvement or what to tackle next]


