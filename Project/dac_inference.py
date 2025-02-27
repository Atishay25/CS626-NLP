#!/usr/bin/env/python3
"""Recipe for training an discrete tokens ctc ASR system with librispeech.

Decoding is performed with greedy decoding at validation time.
At test time, beamsearch is used with an optional external language model.

Authors
 * Pooneh Mousavi 2024
"""

import os
import sys
import torch
import torchaudio
import logging
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main, if_main_process
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        # Forward pass
        # Feature extraction and attention pooling
        with torch.no_grad():
            self.hparams.codec.to(self.device).eval()
            tokens, _ = self.hparams.codec(
                wavs.unsqueeze(1), n_quantizers=self.hparams.num_codebooks
            )
        # print("tokens::", tokens)
        # print("shape of tokens::", tokens.shape)
        
        # print("tokens.movedim(-2, -1)::", tokens.movedim(-2, -1))
        # print("shape of tokens::", tokens.movedim(-2, -1).shape)
        
        embeddings = self.modules.discrete_embedding_layer(
            tokens.movedim(-2, -1)
        )

        # print("embeddings::", embeddings)
        # print("shape of embeddings::", embeddings.shape)

        att_w = self.modules.attention_mlp(embeddings)

        # print("shape of att_w::", att_w.shape)
        #EXPERIMENTS varying num of RVQs used for ASR
        exp_RVQs = 2
        mod_att_w = att_w[:, :, :exp_RVQs, :]
        
        #check att_w (conc on first 4 RVQs, where is 90% of the attention concentrated)
        
        # print("shape of att_w.transpose::", att_w.transpose(2, -1).shape)
        
        # print("shape of mod_att_w.transpose::", mod_att_w.transpose(2, -1).shape)
        mod_embs = embeddings[:, :, :exp_RVQs, :]
        # print("shape of mod_embs::", mod_embs.shape)
        # print("shape of mod_att_w::", mod_att_w.shape)
        # print("shape of matmul::", torch.matmul(att_w.transpose(2, -1), embeddings).shape)
        # feats = torch.matmul(att_w.transpose(2, -1), embeddings).squeeze(-2)
        
        feats = torch.matmul(mod_att_w.transpose(2, -1), mod_embs).squeeze(-2)
        # print("feats::", feats)
        # print("shape of feats::", feats.shape)
        y = self.modules.enc(feats)
        y = y[0]  # As it is an RNN output
        # Compute outputs
        p_tokens = None
        logits = self.modules.ctc_lin(y)
        p_ctc = self.hparams.log_softmax(logits)
        if stage == sb.Stage.VALID:
            p_tokens = sb.decoders.ctc_greedy_decode(
                p_ctc, wav_lens, blank_id=self.hparams.blank_index
            )
        elif stage == sb.Stage.TEST:
            print("WAV LENS", p_ctc, wav_lens)
            p_tokens = test_searcher(p_ctc, wav_lens)

        return p_ctc, wav_lens, p_tokens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        p_ctc, wav_lens, predicted_tokens = predictions
        ids = batch.id
        tokens, tokens_lens = batch.tokens
        loss = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)

        if stage == sb.Stage.VALID:
            # Decode token terms to words
            predicted_words = [
                "".join(self.tokenizer.decode_ndim(utt_seq)).split(" ")
                for utt_seq in predicted_tokens
            ]
        elif stage == sb.Stage.TEST:
            predicted_words = [
                hyp[0].text.split(" ") for hyp in predicted_tokens
            ]

        if stage != sb.Stage.TRAIN:
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_model(
                stage_stats["loss"]
            )
            # old_lr_weights, new_lr_weights = self.hparams.lr_annealing_weights(
            #     stage_stats["loss"]
            # )
            sb.nnet.schedulers.update_learning_rate(
                self.model_optimizer, new_lr_model
            )
            # sb.nnet.schedulers.update_learning_rate(
            #     self.weights_optimizer, new_lr_weights
            # )

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr_model": old_lr_model},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w") as w:
                    self.wer_metric.write_stats(w)

    def init_optimizers(self):
        # "Initializes the weights optimizer and model optimizer"
        # self.weights_optimizer = self.hparams.weights_opt_class(
        #     self.hparams.attention_mlp.parameters()
        # )
        self.model_optimizer = self.hparams.model_opt_class(
            self.hparams.model.parameters()
        )
        self.optimizers_dict = {
            # "weights_optimizer": self.weights_optimizer,
            "model_optimizer": self.model_optimizer,
        }
        # Initializing the weights
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("modelopt", self.model_optimizer)
            # self.checkpointer.add_recoverable(
            #     "weights_opt", self.weights_optimizer
            # )
        else:
            print("NO CHECKPOINT USED")
    def transcribe_dataset(
        self,
        dataset, # Must be obtained from the dataio_function
        min_key, # We load the model with the lowest WER
        loader_kwargs # opts for the dataloading
    ):
        if not isinstance(dataset, DataLoader):
            loader_kwargs["ckpt_prefix"] = None
            dataset = self.make_dataloader(
                dataset, sb.Stage.TEST, **loader_kwargs
            )


        self.on_evaluate_start(min_key=min_key) # We call the on_evaluate_start that will load the best model
        self.modules.eval() # We set the model to eval mode (remove dropout etc)

        # Now we iterate over the dataset and we simply compute_forward and decode
        with torch.no_grad():

            transcripts = []
            for batch in tqdm(dataset, dynamic_ncols=True):
                # Make sure that your compute_forward returns the predictions !!!
                # In the case of the template, when stage = TEST, a beam search is applied
                # in compute_forward().
                out = self.compute_forward(batch, stage=sb.Stage.TEST)
                p_seq, wav_lens, predicted_tokens = out

                # We go from tokens to words.
                predicted_words = [
                    hyp[0].text.split(" ") for hyp in predicted_tokens
                ]
                transcripts.append(predicted_words)

        return transcripts


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")
    transcribe_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["transcribe_csv"], replacements={"data_root": data_folder},
    )
    transcribe_data = transcribe_data.filtered_sorted(sort_key="duration")
    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()] + [transcribe_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        info = torchaudio.info(wav)
        resampled = torchaudio.transforms.Resample(
            info.sample_rate, hparams["sample_rate"],
        )(sig)
        #         resampled = resampled.unsqueeze(0)
        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list", "tokens_list", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = label_encoder.encode_sequence(char_list)
        yield tokens_list
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    special_labels = {
        "blank_label": hparams["blank_index"],
        "unk_label": hparams["unk_index"],
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="char_list",
        special_labels=special_labels,
        sequence_input=True,
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "char_list", "tokens"],
    )
    return train_data, valid_data, test_datasets, transcribe_data, label_encoder


if __name__ == "__main__":

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing Librispeech)
    from librispeech_prepare import prepare_librispeech  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets, transcribe_data, label_encoder = dataio_prepare(
        hparams
    )

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.init_optimizers()
    # Loading the SSL model
    # We dynamicaly add the tokenizer to our brain class.
    asr_brain.tokenizer = label_encoder

    ind2lab = label_encoder.ind2lab
    vocab_list = [ind2lab[x] for x in range(len(ind2lab))]

    from speechbrain.decoders.ctc import CTCBeamSearcher

    test_searcher = CTCBeamSearcher(
        **hparams["test_beam_search"], vocab_list=vocab_list,
    )

    # Training
    #asr_brain.fit(
    #    asr_brain.hparams.epoch_counter,
    #    train_data,
    #    valid_data,
    #    train_loader_kwargs=hparams["train_dataloader_opts"],
    #    valid_loader_kwargs=hparams["valid_dataloader_opts"],
    #)

    # Testing
    #if not os.path.exists(hparams["output_wer_folder"]):
    #    os.makedirs(hparams["output_wer_folder"])

    #for k in test_datasets.keys():  # keys are test_clean, test_other etc
    #    asr_brain.hparams.test_wer_file = os.path.join(
    #        hparams["output_wer_folder"], f"wer_{k}.txt"
    #    )
    #    asr_brain.evaluate(
    #        test_datasets[k],
    #        test_loader_kwargs=hparams["test_dataloader_opts"],
    #        min_key="WER",
    #    )
    transcripts = asr_brain.transcribe_dataset(
        dataset=transcribe_data, # Must be obtained from the dataio_function
        min_key="WER", # We load the model with the lowest WER
        loader_kwargs=hparams["transcribe_dataloader_opts"], # opts for the dataloading
    )
    print(transcripts)
