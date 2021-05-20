
This repository contains code samples related to the [blog post](https://aws.amazon.com/blogs/) Analyzing Multimodal Health Data on AWS.
## Analyzing Multimodal Health Data on AWS

You can use these artifacts to recreate the pipelines and analysis presented in the blog post, and shown below.  

![Architecture on AWS](./images/architecture.png)

## Project Structure

Artifacts for processing each data modality are located in corresponding subdirectories of this repo.  

```
./
./genomics/ <-- Artifacts for genomics pipeline
./imaging/  <-- Artifacts for medical imaging pipeline
./model-train-test/ <-- Artifacts for performing model training and testing
```

There is no directory for the clinical data modality because the blog post demonstrated processing clinical data with Amazon SageMaker Data Wrangler.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.