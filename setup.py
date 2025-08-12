from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

## edit below variables as per your requirements -
REPO_NAME = "News Recommender System"
AUTHOR_USER_NAME = "kanewyp"
SRC_REPO = "news_recommender_system"
LIST_OF_REQUIREMENTS = []


setup(
    name=SRC_REPO,
    version="0.0.1",
    author="kanewyp",
    description="A local packages for CNN based news recommendations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kanewyp/end2end-news-recommendation-system",
    author_email="kanewang630714@gmail.com",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.8",
    install_requires=LIST_OF_REQUIREMENTS
)