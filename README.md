<h2>Twitter Tweets Summarizer and Sentiment Classifier By Jonathan Lampkin</h2>

[![Run on Ainize](https://ainize.ai/images/run_on_ainize_button.svg)](https://ainize.web.app/redirect?git_repo=https://github.com/jonathanlampkin/Common-Computer)


In this repository is the code used to build the backend of the [People's Thoughts](https://main-common-computer-jonathanlampkin.endpoint.ainize.ai/) program deployed on [Ainize](https://ainize.ai/).

<h3>How to use this backend:</h3>

1. git clone https://github.com/jonathanlampkin/Common-Computer-Backend.git

2. cd Common-Computer-Backend

3. docker build --tag {project-name}:{tag} . 

4. docker run -p 8000:8000 {project-name}:{tag} 

<h3>Parameters</h3>

- base_text: Search Query

<h3>Output format</h3>

- {'prediction': {'prediction': Summary, 'sentiment': sentiment classifications (unordered)}}

This program will quickly and simply summarize what twitter users have to say about whatever topic you want and show you how others feel about it.

The program can be found here: https://main-common-computer-jonathanlampkin.endpoint.ainize.ai/

To use the program simply enter any search phrase into the search bar and click Enter.

My blog post covering this program can be found here: https://medium.com/@jmlampkin/my-deployed-ai-model-on-ainize-1bde93a09331

My YouTube video demonstrating this program and explaining the underlying processes can be found here: https://www.youtube.com/watch?v=0-p80wBOGY4

Enjoy!
