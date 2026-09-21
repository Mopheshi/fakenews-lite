# Error Analysis

## False positives against correctly classified legitimate articles

| Property | Among errors | Among correct | Difference | 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| Under 1,000 characters | 36 (18.0%) | 1196 (17.6%) | +0.4% | [-4.4%, +6.4%] | 0.851 |
| Carries a wire dateline | 6 (3.0%) | 4288 (63.0%) | -60.0% | [-62.0%, -56.4%] | 7.2e-73 |
| Armed-conflict vocabulary | 11 (5.5%) | 372 (5.5%) | +0.0% | [-2.4%, +4.1%] | 1 |
| Satire marker | 2 (1.0%) | 29 (0.4%) | +0.6% | [-0.2%, +3.1%] | 0.221 |
| Title mostly upper case | 0 (0.0%) | 0 (0.0%) | +0.0% | [-0.1%, +1.9%] | 1 |
| Three or more exclamation marks | 7 (3.5%) | 116 (1.7%) | +1.8% | [-0.0%, +5.4%] | 0.0892 |
| Ten or more quotation marks | 98 (49.0%) | 2861 (42.0%) | +7.0% | [+0.0%, +13.9%] | 0.0502 |
| Dense first/second person | 38 (19.0%) | 988 (14.5%) | +4.5% | [-0.4%, +10.5%] | 0.0841 |

## False negatives against correctly classified fake articles

| Property | Among errors | Among correct | Difference | 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| Under 1,000 characters | 1 (1.1%) | 1243 (17.0%) | -15.9% | [-17.1%, -11.1%] | 1e-06 |
| Carries a wire dateline | 0 (0.0%) | 2 (0.0%) | -0.0% | [-0.1%, +3.9%] | 1 |
| Armed-conflict vocabulary | 4 (4.3%) | 226 (3.1%) | +1.2% | [-1.4%, +7.5%] | 0.537 |
| Satire marker | 0 (0.0%) | 54 (0.7%) | -0.7% | [-1.0%, +3.2%] | 1 |
| Title mostly upper case | 12 (12.9%) | 825 (11.3%) | +1.6% | [-3.8%, +10.0%] | 0.62 |
| Three or more exclamation marks | 6 (6.5%) | 769 (10.5%) | -4.0% | [-7.6%, +2.9%] | 0.236 |
| Ten or more quotation marks | 33 (35.5%) | 977 (13.3%) | +22.1% | [+13.1%, +32.3%] | 7.06e-08 |
| Dense first/second person | 17 (18.3%) | 2595 (35.4%) | -17.1% | [-23.8%, -8.0%] | 0.000418 |

## Manual categorisation of the read sample

| Error type | Category | Articles | Share of sample |
| --- | --- | --- | --- |
| false positive | unlabelled | 27 | 1.0000 |
| false negative | unlabelled | 13 | 1.0000 |

## Error rate by article provenance

| Group | Articles | Errors | Error rate |
| --- | --- | --- | --- |
| Legitimate, wire dateline | 4294 | 6 | 0.0014 |
| Legitimate, no dateline | 2712 | 194 | 0.0715 |
| Fake, no dateline | 7417 | 93 | 0.0125 |

## Model score by group

| Group | Articles | Mean P(fake) | Median P(fake) | Within 0.1 of the threshold |
| --- | --- | --- | --- | --- |
| False positives | 200 | 0.7126 | 0.6977 | 0.3300 |
| Correct legitimate | 6806 | 0.0377 | 0.0039 | 0.0170 |
| False negatives | 93 | 0.3572 | 0.3961 | 0.4731 |
| Correct fake | 7326 | 0.9646 | 0.9941 | 0.0126 |
