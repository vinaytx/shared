how does the sw check if user is logged into genie? I am logged into genie and the SW should look for tab "Genie One - Databricks" and send a test message to validate that it can get data

Adjusting the tab check to target the Genie space.
Adjusting the tab check to target the Genie space.
Currently it doesn't look at the Genie page at all. probe_session() finds any tab whose URL starts with your workspace URL, runs fetch("/api/2.0/preview/scim/v2/Me") inside it, and treats HTTP 200 (which returns your username) as "logged in". That verifies the Databricks session, but it doesn't prove Genie itself is reachable, and it picks the first workspace tab rather than your Genie tab.

I'll change it to do what you describe: locate the tab by title, then validate with a real Genie round-trip.

Ran a command, shared files
The check now works in two layers, and both are visible from check_browser.py:

Tab selection (browser/chrome.py). The driver lists every open tab and picks, in order: the tab whose title contains browser.genie_tab_title (default "Genie One - Databricks", editable in settings.yaml); otherwise any tab on the workspace URL; otherwise it opens the Genie space URL in a new tab of the same window. It logs which one it chose, and the checker prints all open tab titles so you can see why.

Login check (probe_session). In that tab it runs fetch("/api/2.0/preview/scim/v2/Me") with the tab's cookies. HTTP 200 returns your username, which is what "logged in" means. This is cheap (fraction of a second) and runs before every Genie/SQL call.

Data check (health_check(deep=True)). Sends browser.test_question to Genie through the Conversation API from inside that tab, waits for COMPLETED, and returns the answer text, the SQL Genie generated, and the chat URL. That proves the space ID, your permissions, and the whole Genie path. It's exposed three ways:

python scripts\check_browser.py --test        # from the shell
python main.py --check                        # login only; prints which tab it used
check_databricks_session(deep=true)           # as an MCP tool inside Gemini CLI
Sample output once everything is right:

[3] PASS  Tab 'Genie One - Databricks' found  -> using 'Genie One - Databricks'
[4] PASS  Logged in to Databricks             -> you@company.com
[5] sending test message to Genie: 'How many tables can you query ...' ...
[5] PASS  Genie answered the test message     -> 7
      sql: (none)
      url: https://<ws>/genie/rooms/<id>/chats/<conversation>
Open that URL in the Genie tab and you'll see the test question in the conversation list, which confirms the agent and the window are working on the same space. The test question is configurable, so if your space has a well-known table, a question like "how many rows are in <table>?" gives a stronger check because it also exercises SQL generation.
