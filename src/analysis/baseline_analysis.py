# Import packages
from gensim.models import LdaModel
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

# Import data
data = pd.read_csv(
    "./data/optimization/full/baseline/baseline_performances.csv")

colors = ['tab:blue', 'tab:orange', 'tab:green',
          'tab:red', 'tab:purple', 'tab:brown']


def plot_coherence_comparisons():
    # Get coherence score
    coherences = data.coherence.to_list()
    genres = data.genre.to_list()

    plt.bar(genres, coherences, color=colors)
    plt.title("Baseline Coherence by Genre")
    plt.xlabel("Genre")
    plt.ylabel("Coherence Score")
    plt.savefig("./data/analysis/full/baseline/baseline_coherence.png")
    plt.show()


def plot_perplexity_comparisons():
    perplexities = data.perplexity.to_list()
    genres = data.genre.to_list()

    plt.bar(genres, perplexities, color=colors)
    plt.title("Baseline Perplexity by Genre")
    plt.xlabel("Genre")
    plt.ylabel("Perplexity Score")
    plt.savefig(
        "./data/analysis/full/baseline/baseline_perplexity.png")
    plt.show()


def plot_compute_time_comparisons():
    times = data.time.to_list()
    genres = data.genre.to_list()

    plt.bar(genres, times, color=colors)
    plt.title("Baseline Computation Times by Genre")
    plt.xlabel("Genre")
    plt.ylabel("Computation Time (Seconds)")
    plt.savefig("./data/analysis/full/baseline/baseline_computation_time.png")
    plt.show()


def topic_wordclouds():
    # Load models
    country_model = LdaModel.load(
        "./models/full/baseline/country_baseline.model")
    rap_model = LdaModel.load(
        "./models/full/baseline/rap_baseline.model")
    rb_model = LdaModel.load(
        "./models/full/baseline/rb_baseline.model")
    pop_model = LdaModel.load(
        "./models/full/baseline/pop_baseline.model")
    rock_model = LdaModel.load(
        "./models/full/baseline/rock_baseline.model")

    # Get topics
    country_topics = country_model.show_topics(
        num_topics=4, num_words=50, formatted=False)
    rap_topics = rap_model.show_topics(
        num_topics=4, num_words=50, formatted=False)
    rb_topics = rb_model.show_topics(
        num_topics=4, num_words=50, formatted=False)
    pop_topics = pop_model.show_topics(
        num_topics=4, num_words=50, formatted=False)
    rock_topics = rock_model.show_topics(
        num_topics=4, num_words=50, formatted=False)

    # Make word clouds

    fig, axes = plt.subplots(5, 4, figsize=(15, 5 * 5))

    topics = []
    colors = ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples']

    def get_topics(genre, genre_topics, row):
        print(f"NUM {genre} TOPICS: {len(genre_topics)}")
        for i, topic in enumerate(genre_topics):
            dict_words = dict(topic[1])

            wordcloud = WordCloud(width=400, height=400,
                                  background_color='black',
                                  colormap=colors[row],
                                  min_font_size=10).generate_from_frequencies(dict_words)

            axes[row, i].imshow(wordcloud, interpolation='bilinear')
            axes[row, i].axis("off")
            axes[row, i].set_title(f"{genre} Topic #{i+1}")

        return [{'genre': genre, 'topic': i+1, 'word': word, "frequency": freq} for word, freq in topic[1]]

    topics.extend(get_topics("country", country_topics, 0))
    topics.extend(get_topics("rap", rap_topics, 1))
    topics.extend(get_topics("rb", rb_topics, 2))
    topics.extend(get_topics("pop", pop_topics, 3))
    topics.extend(get_topics("rock", rock_topics, 4))
    print(len(topics[0]))
    print(topics)
    fig.suptitle(
        "Top 4 Topics Per Genre & Meta Baseline Models (Top 50 Terms Per Topic)", y=0.9)
    plt.savefig(
        "./data/analysis/full/baseline/baseline_genre_topics.png")
    plt.show()

    topics_df = pd.DataFrame(topics)
    topics_df.to_csv("./data/analysis/full/baseline/baseline_genre_topics.csv")


if __name__ == "__main__":
    plot_coherence_comparisons()
    plot_perplexity_comparisons()
    plot_compute_time_comparisons()
    topic_wordclouds()
