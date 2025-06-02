import os
import argparse
import requests
import re
import csv
import openai
import tiktoken
from string import Template
from openai import AzureOpenAI
from openai import ChatCompletion
from github import Github
from litellm import model_cost
from pydantic import BaseModel
from typing import List
from engine.base_engine import LLMEngine

def query_github_graphql(query, token):
    """
    Query the GitHub GraphQL API with the provided query and authentication token.

    Args:
        query (str): The GraphQL query string to execute.
        token (str): The GitHub personal access token for authentication.

    Returns:
        dict: The JSON response from the GitHub GraphQL API.

    Raises:
        Exception: If the API request fails or returns a non-200 status code.
    """
    url = "https://api.github.com/graphql"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    try:
        response = requests.post(url, json={"query": query}, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            print(f"GitHub API response: {response_data}")
            return response_data
        else:
            raise Exception(f"Query failed: {response.status_code}, {response.text}")
    except Exception as e:
        return {}

def get_repositories(keyword, token, writer, csvfile, llm_engine, max_repos):
    """
    Fetch repositories from GitHub GraphQL API based on a given keyword.

    Args:
        keyword (str): The search keyword used to find repositories.
        token (str): The GitHub personal access token for authentication.
        max_repos (int): The maximum number of repositories to retrieve.

    Returns:
        list: A list of dictionaries, each containing repository information and linked papers.
    """
    repositories = []
    after_cursor = None  # To store the cursor for pagination

    while True:
        query = f'''
        {{
          search(query: "{keyword} in:readme (paper OR cite OR citation OR arxiv OR doi in:readme) language:Python stars:>10 ",type: REPOSITORY, first: {max_repos}, after: "{after_cursor if after_cursor else ''}") {{
            repositoryCount
            nodes {{
              ... on Repository {{
                name
                url
                description
                readme: object(expression: "HEAD:README.md") {{
                 ... on Blob {{
                    text
                  }}
                }}
              }}
            }}
            pageInfo {{
              hasNextPage
              endCursor
            }}
          }}
        }}
        '''

        # Make the request to GitHub's GraphQL API
        data = query_github_graphql(query, token)

        # Check if the response has 'data' and 'search' key
        if 'data' not in data or 'search' not in data['data']:
            print(f"Error: No 'search' key in the response. Response: {data}")
            break
        total_repositories = data.get('data', {}).get('search', {}).get('repositoryCount', 0)
        print(f"Total repositories matching the search query: {total_repositories}")
        # Process the repositories from the response
        for repo in data['data']['search']['nodes']:
            if repo['readme'] and repo['readme']['text']:
                readme_text = repo['readme']['text']
                # Filtering should happen here
                filter_result, paper_links, input_tokens, output_tokens = research_paper_filter(readme_text, repo['name'], keyword, llm_engine)
                if filter_result == 'YES':
                    print(f"Appending repo from GitHub: {repo['name']}")
                    writer.writerow({
                        "discipline": keyword,
                        "name": repo['name'],
                        "url": repo['url'],
                        "description": repo['description'],
                        "papers": paper_links,
                        "source": "GitHub"
                    })
                    csvfile.flush()
                    print(f"Repository appended! {repo['name']}")

                    repositories.append(repo)

        # Check if there's another page of results
        page_info = data['data']['search']['pageInfo']
        if page_info['hasNextPage']:
            after_cursor = page_info['endCursor']
            print(f"Fetching next page with cursor: {after_cursor}")
        else:
            print("No more pages to fetch.")
            break

def get_github_repo_language(repo_url, token):
    """
    Queries the GitHub API to get the primary language of a repository and returns the repo object.

    Args:
        repo_url (str): The URL of the GitHub repository.
        token (str): The GitHub personal access token.

    Returns:
        tuple: (str, dict) - The primary programming language of the repository (or None if not found),
               and the full repository object as a dictionary (or None if the request fails).
    """
    if "github.com/" not in repo_url:
        return None, None  # Not a GitHub repo
    try:
        repo_name = repo_url.split("github.com/")[-1]  # Extract repo path (owner/repo)
        api_url = f"https://api.github.com/repos/{repo_name}"
        headers = {"Authorization": f"Bearer {token}"}


        response = requests.get(api_url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return data.get("language"), data  # Return both language and full repo details
        else:
            return None, None
    except Exception as e:
        return None, None

def extract_repo_details(repo_data, token):
    """
    Extracts the repository name and README content from a GitHub repo data object.

    Args:
        repo_data (dict): The full repository metadata returned by the GitHub API.
        token (str): GitHub personal access token for authentication.

    Returns:
        tuple: (str, str) - The repository name and README content (or None if not found).
    """
    if not repo_data:
        return None, None  # No data available

    repo_name = repo_data.get("full_name")  # Extract repository name (e.g., "owner/repo")

    # Get the README file from the repository
    try:
        readme_url = repo_data.get("url") + "/readme"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github.v3.raw"}
        response = requests.get(readme_url, headers=headers)
        if response.status_code == 200:
            readme_content = response.text  # Extract README content as text
        else:
            readme_content = None
    except Exception as e:
        readme_content = None

    return repo_name, readme_content

def search_papers_with_code(keyword, writer, github_token, llm_engine, csvfile):
    """
    Search for papers related to a given keyword using the Papers with Code API,
    ensuring that only Python-based repositories are included.

    Args:
        keyword (str): The search keyword used to find papers.
        writer (csv.DictWriter): CSV writer to save results.
        github_token (str): GitHub token to authenticate API requests.

    Returns:
        list: A list of dictionaries, each containing paper information.
    """
    url = f"https://paperswithcode.com/api/v1/search/?q={keyword}"
    response = requests.get(url)

    if response.status_code != 200:
        return []

    papers_data = response.json().get('results', [])

    for entry in papers_data:
        try:
            paper = entry.get('paper', {})
            repository = entry.get('repository', {'name': 'No repository', 'url': '', 'description': 'No description'})

            repo_url = repository.get('url', '')
            # Ensure repository is from GitHub and is Python-based
            if "github.com/" in repo_url:
                repo_language, repo_data = get_github_repo_language(repo_url, github_token)
                if repo_language and repo_language.lower() == "python":
                  #add extra call to filtering function
                    repo_name, readme_content = extract_repo_details(repo_data, github_token)
                    filter_result, paper_links, input_tokens, output_tokens = research_paper_filter(readme_content, repo_name, keyword, llm_engine)
                    if filter_result != 'YES':
                      continue
                    else:
                      writer.writerow({
                          "discipline": keyword,
                          "name": repository.get('name', 'No repository'),
                          "url": repo_url,
                          "description": repository.get('description', 'No description available'),
                          "papers": [paper.get('url_abs', 'No paper URL available')],
                          "source": "Papers with Code"
                      })
                      csvfile.flush()
                      print(f"Repository appended! {repository.get('name', 'No repository')}")
        except Exception as e:
            continue


def research_paper_filter(readme_text, name, keyword, llm_engine):

    """
    Use an LLM to determine if the README text indicates that the repository hosts code related to a research paper
    and extract the relevant paper links.

    Args:
        text (str): The README text to analyze.

    Returns:
        list: A list of unique links to papers (e.g., DOIs, arXiv links, or generic URLs) if the repository
        is confirmed to be related to a research paper.
    """
    class responseFormat (BaseModel):
      research: str
      paper_link: List[str]

    print(f"Extracting links from repo: {name}")
    encoder = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoder.encode(readme_text)
    token_length  = len(tokens)
    print(f"Length of README.md is: {token_length}")
    if token_length < 8000:
        truncated_readme_text = readme_text
    else:
        truncated_readme_text = encoder.decode(tokens[:8000])
    try:
        template = Template("""You are an expert at reading GitHub README.md files thoroughly and determining whether the repository hosts code related to a research paper or not, and you are also skilled at correctly extracting the link to the related paper.
        Think before you respond. Your answer should be based on your thorough understanding of the content of the README.md file. Does the README.md file indicate that the repository hosts code related to a research paper in the discipline of $keyword ? Answer by 'YES' or 'NO' in the research field. If your answer to the previous question is 'YES', extract the link to the related research paper. Make sure to extract the link to the research paper that this repository implements only, this should  be the link to the paper that people would cite if they used the code in the repository for their work, ignoring all other irrelevant links that might be referenced in the README file. Put the link(s) in the field paper_link as a list of links. $truncated_readme_text""")
        prompt = template.substitute(keyword=keyword, truncated_readme_text=truncated_readme_text)
        user_input = [{"role": "user", "content": prompt}]
    
        completion_content, input_tokens, output_tokens = llm_engine.respond_structured(user_input, temperature=0.1, struct_format=responseFormat, top_p=0.9, max_tokens=5000)

        if completion_content.research == "YES":
            print("Repo satisfied criteria, returning paper links")
            return "YES", completion_content.paper_link, input_tokens, output_tokens
        else:
            print("Repo rejected")
            return "NO", [], input_tokens, output_tokens
    except Exception as e:
      return "NO", [], input_tokens, output_tokens

def expand_keywords(base_keywords, llm_engine):
    class ResponseFormat (BaseModel):
        keywords: List[str]
    template = Template("""You are an assistant that generates diverse and related keywords for scientific disciplines. Generate a list of exactly three diverse keywords related to these scientific fields: $base_keywords. Make sure that the generated keywords do not stray away from these scientific disciplines and do not contain broad terms that will confuse the search (e.g. machine learning, algorithms, etc). I would like to use these keywords to retrieve code repositories related to these specific scientific disciplines from GitHub and Papers with Code.""")
    try:
        prompt = template.substitute(base_keywords=", ".join(base_keywords))
        print("Calling LLM to expand keywords...")
        response, _ , _ = llm_engine.respond_structured(prompt, ResponseFormat, temperature=0.5, top_p=0.9, max_tokens=5000) 
        # lines = response.splitlines()
        # cleaned_lines = [re.sub(r'[^a-zA-Z ]', '', line).strip() for line in lines if line.strip()]
        print(f"Response from LLM: {response}")
        expanded_keywords = response.keywords
        print(f"Expanded keywords: {expanded_keywords}")
        return expanded_keywords
    except Exception as e:
        print(f"Error expanding keywords: {e}")
        return base_keywords

def orchestrate_search(base_keywords, github_token, llm_engine, output_csv):
    expanded_keywords = []
    try:
        expanded_keywords = expand_keywords(base_keywords, llm_engine)
    except Exception as e:
        results = []


    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["discipline", "name", "url", "description", "papers", "source"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for keyword in expanded_keywords:
            try:

                # Search GitHub repositories
                get_repositories(keyword, github_token, writer, csvfile, llm_engine, max_repos=100)
                #results.extend(github_repos)
                print(f"Total cost so far: {running_cost}")
                # Search Papers with Code
                search_papers_with_code(keyword, writer, github_token, llm_engine, csvfile)
                #results.extend(papers)
            except Exception as e:
                continue

def main (): 

    parser = argparse.ArgumentParser(description="Crawl repositories from GitHub and Papers with Code.")
    parser.add_argument("--llm_engine_name", type=str, default="azure_gpt-4o", help="Name of the LLM model to use.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to the output JSONL file.", default="../../repo_list/repositories.csv")
    parser.add_argument("--api_version", type=str, default="2024-10-21", help="API version for Azure OpenAI.")
    args = parser.parse_args()

    if "azure_" in args.llm_engine_name:
        api_key = os.environ.get("AZURE_API_KEY")
        azure_endpoint = os.environ.get("AZURE_ENDPOINT")
        api_version = args.api_version
    
    llm_engine = LLMEngine(args.llm_engine_name, api_key=api_key, \
                                azure_endpoint=azure_endpoint, api_version=api_version)
    

    base_keywords = [
        "bioinformatics",
        "psychology",
        "neuroscience",
        "chemistry",
        "geographic information science"
    ]

    # Orchestrate the search process
    orchestrate_search(base_keywords, os.environ.get("GITHUB_TOKEN"), llm_engine, args.output_csv)
if __name__ == "__main__":
    main()