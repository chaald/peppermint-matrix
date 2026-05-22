# Examples

## Technical decision

Prompt:
```
React vs Vue - which fits this project better? Summon the council
```

Steps:
1. Spawn subagents for each configured member (analyst, critic, explorer) in parallel.
2. Collect their individual analyses.
3. Synthesize: the critic may flag React's steeper learning curve, the explorer may suggest Svelte as an alternative. Weigh all perspectives and recommend based on project context.

## Architecture review

Prompt:
```
Let's hear other AIs' opinions on this design
```

Steps:
1. Summarize the design and send it to each council member as a subagent.
2. Collect feedback: the analyst may note structural issues, the critic may find scaling risks, the explorer may propose alternative patterns.
3. Analyze commonalities and synthesize the final review.
