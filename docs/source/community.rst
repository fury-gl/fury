=========
Community
=========

Join Us!
--------

.. raw:: html

    <ul style="list-style-type:none;">
        <li style="display: block"><a href='https://discord.gg/6btFPPj'><i class="fa fa-discord fa-fw"></i> Discord</a></li>
        <li style="display: block"><a href='https://mail.python.org/mailman3/lists/fury.python.org'><i class="fa fa-envelope fa-fw"></i> Mailing list</a></li>
        <li style="display: block"><a href='https://github.com/fury-gl/fury'><i class="fa fa-github fa-fw"></i> Github</a></li>
    <ul>

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