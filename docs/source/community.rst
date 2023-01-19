.. _community:

=========
Community
=========

Join Us!
--------

.. raw:: html
    
    <div class="join-us__container">
        <a href='https://discord.gg/6btFPPj' class="join-us__icon-background m-r-10">
            <i class="fab fa-discord fa-fw join-us__icon"></i>
        </a>
        <a href='https://mail.python.org/mailman3/lists/fury.python.org' class="join-us__icon-background m-r-10">
            <i class="fa fa-envelope fa-fw join-us__icon"></i>
        </a>
        <a href='https://github.com/fury-gl/fury' class="join-us__icon-background m-r-10">
            <i class="fab fa-github fa-fw join-us__icon"></i>
        </a>
    </div>

Contributors
------------

.. raw:: html

    <div id="github_visualization_main_container">
        <div class="github_visualization_visualization_container">
            <div class="github_visualization_basic_stats_container">
                <div class="github_visualization_basic_stats" id="github_visualization_repo_stars">
                    <div class="stat_holder">
                        <div class="basic_stat_icon_holder">
                            <i class="fa-solid fa-star basic_stat_icon"></i>
                        </div>
                        <span><span class="stat-value banner-start-link">{{ basic_stats["stargazers_count"] }}</span> Stars</span>
                        <i class="fa-solid fa-star background_icon"></i>
                    </div>
                </div>
                <div class="github_visualization_basic_stats" id="github_visualization_repo_forks">
                    <div class="stat_holder">
                        <div class="basic_stat_icon_holder">
                            <i class="fa-solid fa-code-fork basic_stat_icon"></i>
                        </div>
                        <span><span class="stat-value">{{ basic_stats["forks_count"] }}</span> Forks<span>
                        <i class="fa-solid fa-code-fork background_icon"></i>
                    </div>
                </div>
                <div class="github_visualization_basic_stats" id="github_visualization_repo_contributors_count">
                    <div class="stat_holder">
                        <div class="basic_stat_icon_holder">
                            <i class="fa-solid fa-users basic_stat_icon"></i>
                        </div>
                        <span><span class="stat-value">{{ contributors["total_contributors"] }}</span> Contributors</span>
                        <i class="fa-solid fa-users background_icon"></i>
                    </div>
                </div>
                <div class="github_visualization_basic_stats" id="github_visualization_repo_commits_count">
                    <div class="stat_holder">
                        <div class="basic_stat_icon_holder">
                            <i class="fa-solid fa-code-commit basic_stat_icon"></i>
                        </div>
                        <span><span class="stat-value">{{ contributors["total_commits"] }}</span> Commits</span>
                        <i class="fa-solid fa-code-commit background_icon"></i>
                    </div>
                </div>
            </div>
        <div id="github_visualization_contributors_wrapper">
        {% for contributor in contributors["contributors"] %}
        <a href="{{ contributor.html_url }}" target="_blank">
        <div class="github_visualization_contributor_info">
            <div class="github_visualization_contributor_img_holder">
                <img class="github_visualization_contributor_img" src="{{ contributor.avatar_url }}">
            </div>
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