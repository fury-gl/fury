=========
Community
=========

Join Us!
--------

- via `Slack <https://join.slack.com/t/fury-gl/shared_invite/enQtNzE1NTk2Mzc3OTQyLTQyNDZiNTUxNWUyZjFmMzZlNDUxZDQ0MzllYjUyYTY1MjFhMmQyYmI3NjJkYzc3YTMwNmRjOWIzMDBjNTYzMDU>`_
- via `mailing list <https://mail.python.org/mailman3/lists/fury.python.org>`_
- via `github <https://github.com/fury-gl/fury>`_

Contributors
------------

.. raw:: html

    <div id="github_visualization_main_container">
        <div class="github_visualization_visualization_container">
            <div class="github_visualization_basic_stats_container">
                <div class="github_visualization_basic_stats" id="github_visualization_repo_stars">
                    <span class="stat-value banner-start-link">{{ basic_stats["stargazers_count"] }}</span> Stars
                    <img class="basic_stat_icon" src="_static/images/stars.png">
                </div>
                <div class="github_visualization_basic_stats" id="github_visualization_repo_forks">
                    <span class="stat-value">{{ basic_stats["forks_count"] }}</span> Forks
                    <img class="basic_stat_icon" src="_static/images/forks.png">
                </div>
                <div class="github_visualization_basic_stats" id="github_visualization_repo_contributors_count">
                    <span class="stat-value">{{ contributors["total_contributors"] }}</span> Contributors
                    <img class="basic_stat_icon" src="_static/images/contributors.png">
                </div>
                <div class="github_visualization_basic_stats" id="github_visualization_repo_commits_count">
                    <span class="stat-value">{{ contributors["total_commits"] }}</span> Commits
                    <img class="basic_stat_icon" src="_static/images/commits.png">
                </div>
            </div>
        <div id="github_visualization_contributors_wrapper">
        {% for contributor in contributors["contributors"] %}
        <a href="{{ contributor.html_url }}" target="_blank">
        <div class="github_visualization_contributor_info">
            <img class="github_visualization_contributor_img" src="{{ contributor.avatar_url }}">
            {% if contributor.fullname %}
            <span class="github_visualization_contributor_name">{{ contributor.fullname }}</span>
            {% else %}
            <span class="github_visualization_contributor_name">{{ contributor.username }}</span>
            {% endif %}
            <span class="github_visualization_contributor_commits">Commits: {{ contributor.nb_commits }}</span>
            <span class="github_visualization_contributor_additions"> ++{{ contributor.total_additions }}</span>
            <span class="github_visualization_contributor_deletions"> --{{contributor.total_deletions }}</span>
        </div>
        </a>
        {% endfor %}
            </div>
        </div>
    </div>