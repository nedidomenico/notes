# inrev

This repository provides utilities for quickly creating merge requests in gitlab, and updating associated
tickets as "In Review". It also provides the ability to merge a PR from the command line. 

## Using

Suppose you are about to start working on ticket PROJ-123, which is to add a newline to the README.
The recommended workflow is as follows:
1. `git wipit3 PROJ-123 addNewline`. This puts you on a branch called `$(whoami)_123-addNewline`,
    and updates the ticket in JIRA.
2. Do the work. Commit early and often. Push regularly (at least before you leave for the day).
3. When it's ready for review, `git inrev3 --from-wipit 'PROJ-123 Add newline to README' --assignee jdoe`.
    Now you're on an `fcr_$(whoami)_123-addNewline` branch, with everything from the `$(whoami)_123-addNewline`
    branch merged into it, and you are one commit ahead of the master branch. There's a PR in gitlab,
    assigned to the user `jdoe` (typically the usernames are our LDAP usernames). If your project
    has a merge request template, it has populated the merge request description.
4. "JSmith" will then go review, you can push more changes to the PR branch, and when it's ready to
    merge, either click the merge button in gitlab, or run `git fmerge3`.

More detail on each step is provided below. All of the inrev functions (`wipit3`, `inrev3`, and `fmerge3`)
take a `-d` or `--debug` argument that provides additional logging as they execute, and an `-h` argument
to show usage information. Additional arguments are described in more detail in what follows, and in
those usage messages.

### Starting a ticket with inrev3

To begin work on a new ticket, checkout the branch you want the work to eventually be merged into
(typically this is `master`). Then run a command like the following:

    git wipit3 ABC-123 fixBug

The first argument to `wipit3` is the full ticket number. You are encouraged to only work on one ticket
at a time, but if you know you are addressing multiple, you can separate ticket numbers with a single space
(thus necessitating that the whole collection is contained in quotes). The second argument is a brief
description of the ticket, just for convenience when browsing a list of branches.

This git alias will:

1. Check that the branch you are on has a remote tracking branch
2. Check that the branch you are on is up-to-date with its remote tracking branch
3. Create a new branch named: `(username)_(ticketnumber)-(desc)`. Note that `(ticketnumber)` will be just
    the number of the ticket, dropping the project name, since it is assumed that the project names are
    closely tied to the repository.
4. Push your new wip branch to the remote as a tracking branch, so you can `git push` as you work.
5. Move the ticket(s) to 'In Progress' in JIRA, with a comment about the branch, and with you as the
    assignee.

### Creating a PR with inrev3

When your branch is ready to be reviewed, you can create a merge request in gitlab with `git inrev3`.
Note that this requires a few things to be true:

1. Your branch name begins with the string "fcr_"
2. Your branch is one commit beyond the branch you want to merge into
3. Your commit message begins with a comma separated list of the tickets that it intends to resolve. For
    example, "IN-23 Fix a bug" or "IN-23,IN-24 Fix a bug and update a README". Note that the commit message
    may include additional lines, but there is no requirement that it have any, or, if it does, what they
    contain.

The default target branch that `inrev3` will assume as the target of your merge, if you don't tell it
otherwise, is `master`. To override this, add another argument to your call, basically
`git inrev3 otherTarget`.

You can also specify the initial assignee for your PR when you create it with `inrev3`. Just pass
`--assignee (gitlab-username)` (e.g., `git inrev3 --assignee jdoe`). Note that most gitlab usernames
are LDAP usernames, which are the first letter of the first name, followed by the last name.

By default, inrev will look for a merge request template file (see
[the wiki](https://atlassian.ccri.com/confluence/display/SFC/Merge+Request+Templates)). If multiple
exist, you will be prompted for the one you wish to use. If you would not like to use a template file,
but some exist, you can pass a `-M` or `--no-mr-template` argument to inrev. If you have multiple template
files and wish to skip the promp, you can specify the file to use with a `-m` or `--mr-template`
argument (e.g., `-m ThisCoolFile.md`). Note that in this case, inrev will search for the file either
as specified, or under the `.gitlab/merge_request_templates/` directory if that file isn't found.
Thus, `-m ThisCoolFile.md` will first look in the current working directory and if `ThisCoolFile.md`
doesn't exist there, it will look for it under the current working directory's
`.gitlab/merge_request_templates` sub-directory.

#### Create a PR with inrev3 from a WIP branch

If you have a work in progress (WIP) branch (either made manually or with the `wipit3` utility described
above), you can use the `--from-wipit` argument to `inrev3` to have it try to create an appropriate
`fcr_` branch for you. This argument expects an additional argument that will become the commit message,
so you would use it like `git inrev3 --from-wipit 'PROJ-123 Solve a hard problem'`.

The `--from-wipit` pathway assumes that you are on the branch that you want to merge. It takes the
target branch as `inrev3` would (i.e., default as `master`, or you can set it with another argument to
`inrev3`), and checks that your local copy of the target branch is up-to-date with the remote and that
your current branch has that target branch merged in, and then creates a new branch on top of the
target branch, and `merge --squash`es your starting (wip) branch into that new branch.

### Merging a PR with fmerge3

If you are on the branch you want to merge, and it has an active PR up, you should be able to merge
the branch with `git fmerge3`. This is equivalent to clicking the "merge" button in gitlab, so if you
prefer that route, it won't break any `inrev` assumptions.

## Setup / Installation

### For inrev users

The recommended way to get set up with inrev (v3) is to add this source directory to your path.
From the current directory, you can do `export PATH=$(pwd):${PATH}`, but it's best to modify your path
in your `~/.bashrc` file, so you don't have to do it regularly.

This repository does come with some git aliases, see the sample [gitconfig](gitconfig) file.
Prior versions of inrev relied on some alises, but this has been replaced with the `PATH`-based approach.
The `PATH`-based approach allows for better command-line argument handling.

`inrev` version 3 has different expectations about the repository and infrastructure setup than prior
versions. Repositories which have not had their infrastructure updated will require the prior version of
inrev. To support multiple inrev versions, we run the new version of the scripts with `git inrev3`, relying
on the `PATH`, and then letting `git inrevgl` (or `git inrev`) co-exist as an alias.

### For repository owners

`inrev` version 3 no longer triggers Jenkins jobs directly on `fmerge`. Now, instead, `fmerge` does exactly
the same thing as clicking the "merge" button in gitlab. It is recommended that you use gitlab webhooks
to trigger Jenkins jobs, or gitlab ci/cd, to check your builds and publish artifacts.

For maven-based projects, there are some generic builder jobs which should work almost out of the box,
and require a little setup in gitlab for a given project to talk to them. More specifically, it is
recommended that you have 2 webhooks registered in your project's "Integrations" settings:

1. One with URL `http://jenkins:8080/project/sfc_genbuilder`, using "Push events" as the only trigger.
2. One with URL `http://jenkins:8080/project/sfc_genmaster`, using "Push events" as the only trigger, this
    time specifying that it only applies to the `master` branch.

The first job will be triggered on all push events, so you'll actually be able to tell if any merge request
builds before you even review it, which is nice. The second job will happen on pushes directly to master,
or on accepted merge requests. In Jenkins, the "genbuilder" basically makes sure a
`mvn clean install` build passes, and then the "genmaster" also does those targets along with
`deploy` so that artifacts go to artifactory. If you believe you have set up webhooks correctly, but are not 
seeing build statuses in gitlab, make sure that the '@jenkinsci' user is a "Developer" under the members in 
your project.

Note that neither the `sfc_genbuilder` nor `sfc_genmaster` build should be used for cutting releases. The
current best recommendation for those is to have your own jenkins job.

A repository owner who wishes to do other things with their builds can copy the "genbuilder" or "genmaster"
jobs in Jenkins to their own jobs, and configure their gitlab webhooks as desired.

### For inrev developers

If you are inspired to develop on the inrev project itself, to add new features or contribute bug fixes,
or just trying to work your way through the code to debug something, please consult the
[DEVELOPING readme](DEVELOPING.md).
