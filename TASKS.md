# Present issues

I suspect leaving the model to figure out the index of the tokens, words and replacements will lead to indexing issues when rendering the diff changelogs in the frontend. This has to be addressed.
Missing a model selector. We aim for provider-agnosticism by design. Bring your own API keys type of deal (or default to the ones I save to .env)
Clicking on edited text should not only display the original, the edit and the rationale for the change but also a button to revert between the two states.

A PR from Devin mentioned that having useCallback and analyze send new API requests on style input refreshes. Worth taking a look
