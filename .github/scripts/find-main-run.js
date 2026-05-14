// Returns the run id of the most recent successful CI run on main, or
// '' if none exists yet. Used to download that run's metrics artifact
// for cross-PR delta comparison.
module.exports = async ({ github, context, core }) => {
  const { data: runs } = await github.rest.actions.listWorkflowRuns({
    owner: context.repo.owner,
    repo: context.repo.repo,
    workflow_id: "ci.yml",
    branch: "main",
    status: "success",
    per_page: 1,
  });
  if (runs.workflow_runs.length === 0) {
    core.warning(
      "No previous successful main run; comment will show absolute values only."
    );
    return "";
  }
  return runs.workflow_runs[0].id.toString();
};
