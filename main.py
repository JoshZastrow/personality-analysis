import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from typing import NamedTuple
from collections import namedtuple as nt
from dataclasses import dataclass
from wordcloud import WordCloud

from components.wordcloud import GroupedColorFunc


@dataclass
class SourceData:
    assessments: pd.DataFrame
    traits: pd.DataFrame
    users: pd.DataFrame
    """
    assessments:
        assessment_id <int>
        domain <str>
        facet <str>
        asessee_id <int>
        assessor_id <int>
        assessee_name <str>
        assessor_name <str>
        rank <float>
        score <float>

    traits:
        trait_id <int>
        domain <str>
        facet <str>
        score <str>
        description <str>

    users:
        user_id <int>
        first_name <str>
        last_name <str>
    """


st.header("Personality Analysis")


@st.cache
def load_data() -> SourceData:

    ASSESSMENT_FPATH = "assets/big_5_data - assessments.csv"
    TRAITS_FPATH = "assets/big_5_data - traits.csv"
    USERS_FPATH = "assets/big_5_data - users.csv"

    assessments = pd.read_csv(ASSESSMENT_FPATH, index_col="assessment_id")
    traits = pd.read_csv(TRAITS_FPATH)
    users = pd.read_csv(USERS_FPATH, index_col="user_id")

    return SourceData(assessments, traits, users)


data = load_data()

users = data.users
assessments = data.assessments
traits = data.traits

del data

person_1 = st.selectbox("Select Candidate", users["first_name"].values)
person_2 = st.selectbox("Select Observer", users["first_name"].values)

col1, col2 = st.columns(2)


def build_wordcloud(assessments: pd.DataFrame, person: str):
    domains = assessments.domain.unique()
    facets = (
        assessments.drop_duplicates(subset=["facet"])
        .loc[lambda d: d.domain != d.facet]
        .loc[:, ["domain", "facet"]]
    )
    colors = ["steelblue", "orange", "silver", "red", "yellow"]
    color_to_words = {
        c: facets.loc[lambda x: x.domain == d, "facet"].to_list()
        for c, d in zip(colors, domains)
    }

    default_color = "grey"
    data = assessments.loc[lambda d: d.domain != d.facet].loc[
        lambda d: d.assessee_name == person
    ]
    word_list = []
    for _, row in data.iterrows():
        word_list.extend([row.facet] * row["rank"])

    np.random.shuffle(word_list)
    text = " ".join(word_list)

    wordcloud = WordCloud().generate(text)
    colorizer = GroupedColorFunc(color_to_words, default_color)
    wordcloud.recolor(color_func=colorizer)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")

    return fig


with col1:
    fig = build_wordcloud(assessments, person_1)
    st.markdown(f"### {person_1}")
    st.write(fig)

with col2:
    fig = build_wordcloud(assessments, person_2)
    st.markdown(f"### {person_2}")
    st.write(fig)


def define_metrics(assessments, person, columns) -> None:
    source_data = assessments.loc[lambda d: d.domain == d.facet].pivot(
        index="domain",
        columns=["assessee_name", "assessor_name"],
        values="rank",
    )
    for idx, row in enumerate(source_data.iterrows()):
        domain = idx
        domain, data = row

        self_review = int(
            data[
                (data.index.get_level_values(0) == person)
                & (data.index.get_level_values(1) == person)
            ].mean()
        )

        peer_review = int(
            data[
                (data.index.get_level_values(0) == person)
                & (data.index.get_level_values(1) != person)
            ].mean()
        )

        delta = self_review - peer_review
        columns[idx].metric(domain, self_review, f"{delta} ({peer_review})")


st.header(f"{person_1}'s Traits")
columns = st.columns(5)
define_metrics(assessments, person_1, columns)

st.header(f"{person_2}'s Traits")
columns = st.columns(5)
define_metrics(assessments, person_2, columns)

# st.markdown(
#     """
#     There are three things to look at with these metrics. The first being
#     how we see ourselves, the second is how we are seen by our other, and the
#     third is how personalities influence our assessments.

#     ### Under Estimating Each Others Extraversion and Open Mindedness
#     One thing that is interesting is that while we both have high levels of
#     extraversion, we both gave ourselves higher scores than what we gave each other.
#     My own score is in the top 1%, 12% points higher than your assessment (87th percentile).

#     Meanwhile, your self-assessment came in at at 78, 40 points higher than my assessment (38).
#     One hypothesis is that we are less extroverted when we are around each other, even though we
#     both feel more extroverted. This dynamic plays out in the Openness to Experience as well. We are both
#     more open to experiences than we give each other credit for.

#     ### How Agreeable and Conscientious we see each other
#     In both traits, I gave higher marks to myself than you did, and I gave you
#     higher marks than you gave yourself. I pegged you at the top 2% of the most
#     agreeable people--16 points higher than your own assessment, and I scored you
#     15 points higher on your own rating of conscientiousness. I scored myself higher
#     in both these areas as well.

#     Perhaps you hold a more realistic perspective on these traits for
#     both you and I. Low morality scores may indicate this perspective could be withheld,
#     leaving me with a higher sense of optimism. Then again, I do have an inherent sense
#     of optimism, a high level of trust combined with dominant extraversion may affect
#     my perception of agreeablness for both of us.

# """
# )


# ## MOST UNIQUE CHARACTERISTICS ###########################
# most_diff = (
#     combined.sort_values("diff")
#     .loc[lambda d: ~(d.domain == d.facet)]
#     .tail(5)
#     .loc[
#         :,
#         ["domain", "facet", "rank_josh", "rank_allie", "description"],
#     ]
# )

# st.table(most_diff)
# ########################################################

# lollipop_data = assessments.merge(traits, on=["domain", "facet", "score"])
# lollipop = alt.layer(data=lollipop_data).transform_filter(
#     filter={"field": "assessee_name", "oneOf": [person_1, person_2]}
# )
# st.table(lollipop_data.head(1))

# for name, group in lollipop_data.groupby(['domain']):

# lollipop += (
#     alt.Chart()
#     .mark_line(color="grey")
#     .encode(
#         x="rank:Q",
#         y=alt.Y(
#             "facet",
#             sort=combined.sort_values("rank_josh", ascending=True).facet.values,
#         ),
#         detail="facet:N",
#         tooltip=["description"],
#     )
# )

# # Add points for life expectancy in 1955 & 2000
# lollipop += (
#     alt.Chart()
#     .mark_point(size=100, opacity=1, filled=True)
#     .properties(width=800)
#     .encode(
#         x="rank",
#         y=alt.Y(
#             "facet",
#             sort=combined.sort_values("rank_josh", ascending=True).facet.values,
#         ),
#         color=alt.Color(
#             "name",
#             scale=alt.Scale(
#                 domain=["Josh", "Allie"], range=["skyblue", "maroon"]
#             ),
#         ),
#         tooltip=["description"],
#     )
#     .interactive()
# )

# # st.table(assessments.head())
# st.altair_chart(lollipop)

# ### RADAR CHARTS ########################################
import plotly.graph_objects as go

domains = assessments.domain.drop_duplicates().to_list()
domain_selection = st.radio("Select Domain:", options=domains)


def build_radar_charts(assessments, person_1, person_2, domain, style="radar"):
    source_data = assessments.loc[lambda d: d.domain == domain]

    # Josh's perspective of Josh
    comp_1 = source_data.loc[
        lambda d: (d.assessee_name == person_1) & (d.assessor_name == person_1)
    ].assign(analysis="{person_1} self review")

    # Allie's perspective of Josh
    comp_2 = source_data.loc[
        lambda d: (d.assessee_name == person_1) & (d.assessor_name == person_2)
    ].assign(analysis="{person_2} peer review")

    # Allie's perspective of Allie
    comp_3 = source_data.loc[
        lambda d: (d.assessee_name == person_2) & (d.assessor_name == person_2)
    ].assign(analysis="{person_2} self review")

    # Josh's perspective of Allie
    comp_4 = source_data.loc[
        lambda d: (d.assessee_name == person_2) & (d.assessor_name == person_1)
    ].assign(analysis="{person_1} peer review")

    comp_1 = comp_1.groupby(["facet"]).mean()["rank"].reset_index()
    comp_2 = comp_2.groupby(["facet"]).mean()["rank"].reset_index()
    comp_3 = comp_3.groupby(["facet"]).mean()["rank"].reset_index()
    comp_4 = comp_4.groupby(["facet"]).mean()["rank"].reset_index()

    if style == "radar":
    fig1 = go.Figure()

    # Perpective of Josh
    fig1.add_trace(
        go.Scatterpolar(
            r=comp_1["rank"],
            theta=comp_1["facet"],
            fill="toself",
            name=f"{person_1}'s review ",
        )
    )
    fig1.add_trace(
        go.Scatterpolar(
            r=comp_3["rank"],
            theta=comp_3["facet"],
            fill="toself",
            name=f"{person_2}'s review",
        )
    )

    fig1.update_layout(
        title=dict(text=f"How We See Ourselves"),
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
    )

    fig2 = go.Figure()

    fig2.add_trace(
        go.Scatterpolar(
            r=comp_2["rank"],
            theta=comp_2["facet"],
            fill="toself",
            name=f"{person_1}",
        )
    )
    fig2.add_trace(
        go.Scatterpolar(
            r=comp_4["rank"],
            theta=comp_4["facet"],
            fill="toself",
            name=f"{person_2}",
        )
    )

    fig2.update_layout(
        title=dict(text=f"How We See Each other"),
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
    )

    fig3 = go.Figure()

    fig3.add_trace(
        go.Scatterpolar(
            r=comp_1["rank"],
            theta=comp_1["facet"],
            fill="toself",
            name=f"{person_1}",
        )
    )
    fig3.add_trace(
        go.Scatterpolar(
            r=comp_4["rank"],
            theta=comp_4["facet"],
            fill="toself",
            name=f"{person_2}",
        )
    )

    fig3.update_layout(
        title=dict(text=f"How {person_1} sees us"),
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
    )

    fig4 = go.Figure()

    fig4.add_trace(
        go.Scatterpolar(
            r=comp_2["rank"],
            theta=comp_2["facet"],
            fill="toself",
            name=f"{person_1}",
        )
    )
    fig4.add_trace(
        go.Scatterpolar(
            r=comp_3["rank"],
            theta=comp_3["facet"],
            fill="toself",
            name=f"{person_2}",
        )
    )

    fig4.update_layout(
        title=dict(text=f"How {person_2} sees us"),
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
    )

    fig5 = go.Figure()

    fig5.add_trace(
        go.Scatterpolar(
            r=comp_1["rank"],
            theta=comp_1["facet"],
            fill="toself",
            name=f"{person_1}",
        )
    )
    fig5.add_trace(
        go.Scatterpolar(
            r=comp_2["rank"],
            theta=comp_2["facet"],
            fill="toself",
            name=f"{person_2}",
        )
    )

    fig5.update_layout(
        title=dict(text=f"Who knows {person_1} best?"),
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
    )

    fig6 = go.Figure()

    fig6.add_trace(
        go.Scatterpolar(
            r=comp_3["rank"],
            theta=comp_3["facet"],
            fill="toself",
            name=f"{person_1}",
        )
    )
    fig6.add_trace(
        go.Scatterpolar(
            r=comp_4["rank"],
            theta=comp_4["facet"],
            fill="toself",
            name=f"{person_2}",
        )
    )

    fig6.update_layout(
        title=dict(text=f"Who knows {person_2} best?"),
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
    )

    return fig1, fig2, fig3, fig4, fig5, fig6


fig1, fig2, fig3, fig4, fig5, fig6 = build_radar_charts(
    assessments, person_1, person_2, domain_selection
)
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.plotly_chart(fig4, use_container_width=True)


col5, col6 = st.columns(2)

with col3:
    st.plotly_chart(fig5, use_container_width=True)

with col4:
    st.plotly_chart(fig6, use_container_width=True)


def build_comp_table(assessments, person_1, person_2) -> None:
    source_data = (
        assessments.loc[lambda d: d.domain != d.facet]
        .loc[lambda d: d.assessee_name == d.assessor_name]
        .pivot(
            index="facet",
            columns=["assessee_name"],
            values="rank",
        )
        .assign(diff=lambda x: abs(x[person_1] - x[person_2]))
        .sort_values("diff", ascending=False)
        .reset_index()
        # .drop(["diff"], axis=1)
    )

    return source_data


data = assessments.loc[lambda d: d.domain != d.facet].loc[
    lambda d: d.assessee_name == d.assessor_name
]
comp = build_comp_table(assessments, person_1, person_2)

for name, group in assessments.groupby("domain"):
    source = group.loc[lambda d: d.assessor_name == person_2].loc[
        :, ["assessee_name", "facet", "rank"]
    ]
    # st.table(source)

    bars = (
        alt.Chart(source)
        .mark_bar()
        .encode(
            y="sum(rank):Q",
            x=alt.X("assessee_name:O", title="name"),
            color=alt.Color("assessee_name:N", title="test"),
            column=alt.Row("facet:N", sort="descending"),
        )
        .configure_axisX(labelColor="white", tickColor="white")
        .configure_header(
            titleColor="white",
            titleFontSize=14,
            labelColor="white",
            labelFontSize=10,
        )
    )

    st.altair_chart(bars)
