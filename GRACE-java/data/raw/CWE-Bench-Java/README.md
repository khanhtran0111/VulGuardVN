---
configs:
  - config_name: advisory
    data_files: "data/advisory/*.arrow"
  - config_name: project_info
    data_files: "data/project_info/*.arrow"
  - config_name: build_info
    data_files: "data/build_info/*.arrow"
  - config_name: fix_info
    data_files: "data/fix_info/*.arrow"
license: mit
language:
- en
size_categories:
- n<1K
tags:
- code
- cybersecurity
- vulnerability
---

# CWE-Bench-Java

<!-- Provide a quick summary of the dataset. -->

This repository contains the dataset CWE-Bench-Java presented in the paper [LLM-Assisted Static Analysis for Detecting Security Vulnerabilities](https://arxiv.org/abs/2405.17238). At a high level, this dataset contains 120 CVEs spanning 4 CWEs, namely path-traversal, OS-command injection, cross-site scripting, and code-injection. Each CVE includes the buggy and fixed source code of the project, along with the information of the fixed files and functions. We provide the seed information for each CVE in this repository, as well as advisories.

**CWE-Bench-Java Github Repository** (https://github.com/iris-sast/cwe-bench-java) - The CWE-Bench-Java repository contains more details about the benchmark, and how to reproduce the benchmark for the paper. For any feedback, please open an issue.

**IRIS Paper Github Repository** (https://github.com/iris-sast/iris) - The IRIS repository contains instructions on reproducing the paper. 

## Dataset Details

The `raw_data` directory contains the data files from the Github repository.

The `data` directory contains the contents of `raw_data`, processed for usage with the Hugging Face datasets library. Includes Arrow-formatted files. 

### Project Identifier

In this dataset, each project is uniquely identified with a **Project Slug**, encompassing its repository name, CVE ID, and a tag corresponding to the buggy version of the project.
We show one example below:

```
DSpace__DSpace_CVE-2016-10726_4.4
^^^^^^  ^^^^^^ ^^^^^^^^^^^^^^ ^^^
|       |      |              |--> Version Tag
|       |      |--> CVE ID
|       |--> Repository name
|--> Github Username
```

All the patches, advisory information, build information, and fix information are associated with project slugs.
Since there are 120 projects in the CWE-Bench-Java dataset, we have 120 unique project slugs.
Note that a single repository may be found to have different CVEs in different versions.

### Packaged Data

```
- data/
  - project_info.csv
  - build_info.csv
  - fix_info.csv
- advisory/<project_slug>.json
```

The core set of information in this dataset lies in two files, `data/project_info.csv` and `data/fix_info.csv`.
We also provide other essential information such as CVE advisory, and build information for the projects.
We now go into the project information and fix information CSVs.

### Advisory 

vuln_id | schema_version | published_date | modified_date | aliases | summary | details | cvss_version | cvss_vector | cvss_score | severity_rating | cwe_ids | ecosystem | package_name | introduced_version | fixed_version | references | github_reviewed | github_reviewed_at | nvd_published_at
--------|----------------|----------------|---------------|---------|---------|---------|---------------|--------------|-------------|------------------|---------|-----------|----------------|--------------------|---------------|------------|------------------|---------------------|------------------
CVE-2022-12345 | 1.4.0 | 2022-01-01 | 2022-01-10 | GHSA-xxxx-yyyy-zzzz | Example XSS vuln | Reflected XSS in parameter `q` | CVSS:3.1 | AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N | 6.1 | MODERATE | CWE-79 | Maven | com.example:library | 1.0.0 | 1.0.1 | https://example.com/advisory | true | 2022-01-05T12:00:00Z | 2022-01-01T00:00:00Z

This data is extracted from the CWE-Mitre net database and converted to JSON format.
We now get into each field and explain what they are.
- `vuln_id`: a string like `CVE-2021-44667` or `GHSA-xxxx-yyyy-zzzz` representing the unique ID of the vulnerability.
- `schema_version`: a string indicating the schema used to encode this data (e.g., `"1.4.0"`).
- `published_date`: a date (in ISO 8601 format, e.g. `2022-03-12`) representing when the vulnerability was first disclosed.
- `modified_date`: a date representing when the record was last updated.
- `aliases`: a list of strings (e.g., `[ "CVE-2021-44667" ]`) capturing alternate identifiers.
- `summary`: a short string summarizing the vulnerability in one sentence.
- `details`: a longer string giving a full description of how the vulnerability occurs and its impact.
- `cvss_version`: a string like `CVSS:3.1` indicating the version of the CVSS specification used.
- `cvss_vector`: a CVSS vector string (e.g., `AV:N/AC:L/PR:N/UI:R/S:C/C:L/I:L/A:N`) describing the severity dimensions.
- `cvss_score`: a float between 0.0 and 10.0 quantifying the vulnerability severity.
- `severity_rating`: a string with one of `LOW`, `MODERATE`, `HIGH`, or `CRITICAL` as a qualitative severity label.
- `cwe_ids`: a list of strings like `["CWE-79"]` referring to Common Weakness Enumeration identifiers.
- `ecosystem`: a string such as `"Maven"` or `"npm"` indicating the software package manager ecosystem affected.
- `package_name`: a string like `com.alibaba.nacos:nacos-common` identifying the specific package.
- `introduced_version`: a string denoting the first version where the vulnerability was introduced (e.g., `"0"`).
- `fixed_version`: a string indicating the version where the issue was patched (e.g., `"2.0.4"`).
- `references`: a list of URLs pointing to advisory pages, commits, issue trackers, etc.
- `github_reviewed`: a boolean (`true` or `false`) showing whether GitHub has reviewed this vulnerability.
- `github_reviewed_at`: a timestamp (e.g., `2022-03-14T23:25:35Z`) of when GitHub reviewed the advisory.
- `nvd_published_at`: a timestamp of when the NVD officially published the vulnerability.

### Project Info

| id | project_slug | cve_id | cwe_id | cwe_name | github_username | github_repository_name | github_tag | github_url | advisory_id | buggy_commit_id | fix_commit_ids |
| -- | ------------ | ------ | -------|----------|-----------------|------------------------|------------|------------|-------------|-----------------|----------------|
| 1 | DSpace__DSpace_CVE-2016-10726_4.4 | CVE-2016-10726 | CWE-022 | Path Traversal | DSpace | DSpace | 4.4 | https://github.com/DSpace/DSpace | GHSA-4m9r-5gqp-7j82 | ca4c86b1baa4e0b07975b1da86a34a6e7170b3b7 | 4239abd2dd2ae0dedd7edc95a5c9f264fdcf639d |

Each row in `data/project_info.csv` looks like the example above.
We now get into each field and explain what they are.

- `id`: an integer from 1 to 120
- `project_slug`: (explained in the previous section)
- `cve_id`: a common vulnerability identifier `CVE-XXXX-XXXXX`
- `cwe_id`: a common weakness enumeration (CWE) identifier. In our dataset, there is only `CWE-022`, `CWE-078`, `CWE-079`, `CWE-094`
- `cwe_name`: the name of the CWE
- `github_username`: the user/organization that owns the repository on Github
- `github_repository_name`: the repository name on Github
- `github_tag`: the tag associated with the version where the vulnerability is found; usually a version tag
- `github_url`: the URL to the github repository
- `advisory_id`: the advisory ID in Github Security Advisory database
- `buggy_commit_id`: the commit hash (like `ca4c86b1baa4e0b07975b1da86a34a6e7170b3b7`) where the vulnerability can be reproduced
- `fix_commit_ids`: the set of commit hashes (sequentially ordered and separated with semicolon `;`) corresponding to the fix of the vulnerability

### Fix Info

The `data/fix_info.csv` file contains the fixed Java methods and classes to each CVE.
In general, the fix could span over multiple commits, and a change could be made to arbitrary files in the repository, including resources (like `.txt`, `.html`) and Java source files (including core source code and test cases).
In this table, we only include the methods and classes that are considered core.
Many of the rows in this table is manually vetted and labeled.
Note that there may be fixes on class variables, in which case there will not be method information associated with the fix.
A single function may be "fixed" by multiple commits.

Each row in `data/fix_info.csv` looks like the following.

| project_slug | cve | github_username | github_repository_name | commit | file | class | class_start | class_end | method | method_start | method_end | signature |
|--------------|-----|-----------------|------------------------|--------|------|-------|-------------|-----------|--------|--------------|------------|-----------|
| apache__activemq_CVE-2014-3576_5.10.2 | CVE-2014-3576 | apache | activemq | `00921f22ff9a8792d7663ef8fadd4823402a6324` | `activemq-broker/src/main/java/org/apache/activemq/broker/TransportConnection.java` | `TransportConnection` | 104 | 1655 | `processControlCommand` | 1536 | 1541 | `Response processControlCommand(ControlCommand)` |

- `project_slug`: the unique identifier of each project
- `cve_id`: the CVE id
- `github_username`: the user/organization that owns the repository on Github
- `github_repository_name`: the repository name on Github
- `commit`: the commit hash containing this fix
- `file`: the `.java` file that is fixed
- `class`: the name of the class that is fixed
- `class_start`, `class_end`: the start and end line number of the class
- `method`: the name of the method that is fixed
- `method_start`, `method_end`: the start and end line number of the method
- `signature`: the signature of the method. Note that we might have multiple overloaded methods with the same name but with different signatures


### Dataset Sources

An extension of this dataset can be found on the Github repository, which provides utilities to fetch and build the relevant projects, and a simple website visualizer.

- **Curated by:** Ziyang Li, Saikat Dutta 
- **License:** MIT 

<!-- Provide the basic links for the dataset. -->

- **Repository:** [CWE-Bench-Java](https://github.com/iris-sast/cwe-bench-java/tree/master)
- **Paper [optional]:** [LLM-Assisted Static Analysis for Detecting Security Vulnerabilities](https://arxiv.org/abs/2405.17238)

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

- Study patterns from previous vulnerability fixes to address current vulnerabilities effectively.
- Evaluate and compare performance of static analysis tools (e.g., CodeQL, Semgrep) on real-world vulnerabilities.
- Assess and measure accuracy, recall, precision, and false-positive rates of various security tools across different vulnerability types.
- Use detailed fix information to improve automatic patch generation and vulnerability remediation systems.
- Provide examples of code vulnerabilities and their respective fixes to educate developers, cybersecurity professionals, and students.

### Direct Use Examples

<!-- This section describes suitable use cases for the dataset. -->

- Analyzing past Java-based CVE fixes on CWE-022 classifications to develop guidelines for addressing file-access vulnerabilities
- Conducting controlled experiments to systematically quantify false positives and true positive detections of security tools for injection
- Developing interactive security training modules that showcase vulnerabilities alongside detailed explanations of the actual patches

### Out-of-Scope Use

- Because the dataset covers only specific vulnerability types and limited CVEs, it should not be treated as a complete security benchmark for evaluating the entire security posture of software projects.
- Sole reliance on the provided CVE data without additional context or tooling may lead to misinterpretations on what the vulnerability actually is.
- Tools unrelated to static analysis or vulnerability patching (e.g., antivirus software) would likely see limited benefit from this dataset.

### Misuse and Malicious Use

- Attackers could analyze vulnerable code examples to understand how to exploit similar software weaknesses
- Detailed information about vulnerabilities might aid in crafting targeted attacks against unpatched software versions.

### Curation Rationale

The dataset was created to provide a high-quality benchmark for evaluating the ability of IRIS to detect and fix real-world vulnerabilities in Java code. 
Existing benchmarks often lack direct links to real CVEs and actionable fixes. 
This dataset bridges that gap with reproducible, well-labeled examples tied to CVEs and CWEs.

### Source Data

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->
- GitHub Security Advisories (https://github.com/advisories): Used to extract structured CVE metadata, severity ratings, affected packages, ecosystem information.
- [Github commits](https://docs.github.com/en/pull-requests/committing-changes-to-your-project/creating-and-editing-commits/about-commits): Commit logs and diffs were used to identify the buggy and fixed versions of code and determine class/method-level changes needed to fix the vulnerability.
- MITRE CWE Database (https://cwe.mitre.org): Provided the classification, naming, description, and available links related to each vulnerability type.

#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

- Projects were selected based on the availability of Java source code and documentation on how to fix the error 
- Buggy and fixed code versions using git diff and commit history
- Associated advisory information from GitHub Security Advisories and NVD
- Class and method boundaries using AST analysis 
- Patch validation was performed via manual review and automated testing where possible.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

- Biased toward Java: Only Java-based CVEs are included; generalization to other languages (e.g., C/C++, Python) should be done with caution.
- Limited in CWE scope: Covers only 4 CWEs — CWE-022, CWE-078, CWE-079, CWE-094.
- Biased toward open-source: Enterprise and closed-source vulnerabilities are excluded, limiting the scope of evaluation.
- Manually vetted, which introduces potential human error or subjective judgment in what counts as the “core fix.”

## Citation

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->
Consider citing our paper:

```
@article{li2024iris,
      title={LLM-Assisted Static Analysis for Detecting Security Vulnerabilities},
      author={Ziyang Li and Saikat Dutta and Mayur Naik},
      year={2024},
      eprint={2405.17238},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2405.17238},
}
```

## Glossary

<!-- If relevant, include terms and calculations in this section that can help readers understand the dataset or dataset card. -->

- CVE (Common Vulnerabilities and Exposures): A unique identifier for a known security vulnerability.
- CWE (Common Weakness Enumeration): A formalized taxonomy of software vulnerability types.
- CVSS (Common Vulnerability Scoring System): A standard for assessing the severity of security vulnerabilities.
- Static Analysis: Code analysis technique that examines source code without executing it.
- Patch: A set of changes to a program designed to fix a known issue or vulnerability

## Dataset Card Authors 

- Ziyang Li (University of Pennsylvania)
- Saikat Dutta (Cornell University)
- Mayur Naik (University of Pennsylvania)
- Claire Wang (University of Pennsylvania)
- Kevin Xue (University of Pennsylvania)
- Amartya Das

## Dataset Card Contact

For any feedback, questions, concerns - please [open an issue](https://github.com/iris-sast/cwe-bench-java/issues) on the CWE-Bench-Java Github repository.