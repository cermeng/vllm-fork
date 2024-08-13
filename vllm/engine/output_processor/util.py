from typing import List
from typing import Sequence as GenericSequence
from typing import Union

from vllm.sequence import PoolerOutput, SamplerOutput, SequenceGroupOutput
from vllm.spec_decode.metrics import SpecDecodeWorkerMetrics

def create_output_by_sequence_group(
        outputs: GenericSequence[Union[SamplerOutput, PoolerOutput]],
        num_seq_groups: int) -> List[List[SequenceGroupOutput]]:
    """Helper method which transforms a 2d list organized by
    [step][sequence group] into [sequence group][step].
    """
    output_by_sequence_group: List[List[SequenceGroupOutput]] = [
        [] for _ in range(num_seq_groups)
    ]
    # spec_metrics: List[List[SpecDecodeWorkerMetrics]] = [
    #     [] for _ in range(num_seq_groups)
    # ]

    for step in outputs:
        for i, sequence_group_output in enumerate(step):
            output_by_sequence_group[i].append(sequence_group_output)
            # if sequence_group_output.spec_decode_worker_metrics is not None:
            #     spec_metrics[i].append(
            #         sequence_group_output.spec_decode_worker_metrics)

    return output_by_sequence_group

# def create_spec_metrics_by_sequence_group(
#         outputs: GenericSequence[Union[SamplerOutput, PoolerOutput]],
#         num_seq_groups: int) -> List[SpecDecodeWorkerMetrics]:
#     """Helper method which transforms a 2d list organized by
#     [step][sequence group] into [sequence group][step].
#     """