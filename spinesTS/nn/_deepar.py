"""Likelihood function and class for the DeepAR framework"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules import Module
from torch.distributions.normal import Normal
from spinesTS.base import Device


class DeepAR(nn.Module):
    def __init__(self, input_size, output_size = 1, encoder_len = 1, 
        decoder_len = None, encoder = None, 
        decoder = None, hidden_size = None, num_layers = None, 
        bias = True, dropout = 0, rnn_hidden_size = None,
        rnn_layers = None):
        """
        Creates a DeepAR NN with LSTM encoder and decoder by default.
        -----------------------------------------------------------------------
        
        Necessary parameters in all cases:
        
        :param input_size: The number of features at a single time-step
        :param output_size: The size of the output at a single time-step
        :param encoder_len: The length of the encoder input sequence
        :param decoder_len: The length of the decoder output sequence. Defaults
        to encoder_len.
        
        -----------------------------------------------------------------------
        Encoder and decoder parameters:
            :param encoder: If supplied, then please edit forward() accordingly.
        If None, then a LSTM is used if decoder is also None. Otherwise, an 
        error is thrown.
            :param encoder: If supplied, then please edit forward() accordingly.
        If None, then a LSTM is used if encoder is also None. Otherwise, an 
        error is thrown.
            :param rnn_hidden_size: Supply if using custom encoder/decoder. 
        Otherwise, leave None.
            :param rnn_layers: Supply if using custom encoder/decoder. 
        Otherwise, leave None.
        -----------------------------------------------------------------------
        If the default LSTM is chosen for the encoder and decoder, then the
        following LSTM parameters should be specified:
            :param hidden_size:
            :param num_layers:
            :param bias:
            :param dropout:
        """
        assert not ((encoder == None) ^ (decoder == None)), "Either both or none of encoder and decoder can be None"
            
        decoder_len = encoder_len if (decoder_len == None) else decoder_len

        super(DeepAR, self).__init__()

        # If an encoder and decoder are not supplied, then set them to 
        # LSTMs. There are two cases:
        #   (1) LSTMs are used, and LSTM parameters are supplied
        #   (2) LSTMs are not used, and rnn sizes are supplied
        #
        if (encoder == None and decoder == None):
            assert (hidden_size != None and num_layers != None), "Using LSTM, but hidden_size and num_layers not specified!"
                
            self.encoder = nn.LSTM(input_size = input_size,
                                hidden_size = hidden_size,
                                num_layers = num_layers,
                                bias = bias,
                                batch_first = True,
                                dropout = dropout,
                                bidirectional = False)
            self.decoder = nn.LSTM(input_size = output_size,
                                hidden_size = hidden_size,
                                num_layers = num_layers,
                                bias = bias,
                                batch_first = True,
                                dropout = dropout, 
                                bidirectional = False)
            rnn_hidden_size = hidden_size
            rnn_layers = num_layers
        else:
            assert (rnn_hidden_size != None and rnn_layers != None), "Using custom encoder and decoder without specifying rnn sizes!"
                
            self.encoder = encoder
            self.decoder = decoder

        self.squeeze_layers_mu = nn.Linear(rnn_layers, 1)
        self.squeeze_layers_sig = nn.Linear(rnn_layers, 1)
        self.probability_mean = nn.Linear(rnn_hidden_size, 
            output_size)
        self.probability_std = nn.Linear(rnn_hidden_size,
            output_size)
        self.std_softplus = nn.Softplus()

        # Record parameters
        self.output_size = output_size
        self.rnn_layers = rnn_layers
        self.rnn_hidden_size = rnn_hidden_size
        self.device = Device
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.in_train_mode = True

    def train(self, mode = True):
        """ 
        If training, then pass true target values to decoder. Otherwise,
        pass value sampled from existing distribution.
        """
        super().train(mode)
        self.in_train_mode = mode
        self.encoder.train(mode = mode)
        self.decoder.train(mode = mode)
    
    def eval(self):
        """ 
        If training, then pass true target values to decoder. Otherwise,
        pass value sampled from existing distribution.
        """
        super().eval()
        self.in_train_mode = False
        self.encoder.eval()
        self.decoder.eval()

    def forward(self, input, target, hidden_encoder = None, 
        cell_encoder = None):
        """
        Performs a forward pass.[*]
        [*] - Edit this code if using a custom rnn encoder/decoder or 
        non-Gaussian distribution.
        -----------------------------------------------------------------------
        Requires:
            * For training, `input` and `target` are individual sequences of 
        encoder and decoder length respectively. For testing, `input` and 
        `target` are sequences of encoder length and length 1 respectively.
        It is the responsibility of the caller to appropriately zero pad inputs 
        and targets. The original paper does not pad the targets, only inputs.
        -----------------------------------------------------------------------
        :param input: Input tensor of shape [batch_size, encoder_len, input_size]
        :param target: If training, the ground truth. If testing, the last
        observed target value.
        :return outputs: Tensor of shape [batch_size, decoder_len, output_size]
        :return mu_collection: Tensor of shape [batch_size, decoder_len, 
        output_size]
        :return sigma_collection: Tensor of shape [batch_size, decoder_len, 
        output_size]
        :return hidden_encoder: Tensor of shape [rnn_layers, batch_size, 
        rnn_hidden_size] if training. None if testing.
        :return cell_encoder: Tensor of shape [rnn_layers, batch_size, 
        rnn_hidden_size] if training. None if testing.
        -----------------------------------------------------------------------
        Training Parameters:
    
        :param target: Target output tensor of shape [batch_size, decoder_len, 
            output_size]. 
        :param hidden_encoder: Include for persistent states.
        :param cell_encoder: [SEE ABOVE]
        -----------------------------------------------------------------------
        Testing Parameters:
        :param target: Target output tensor of shape [batch_size, 1, output_size]
        """
        batch_size = input.size()[0]

        # If not provided for training, zero-initialize the encoder's hidden
        # and cell layers
        if (self.in_train_mode and hidden_encoder == None and cell_encoder == None):
            hidden_encoder = torch.zeros(self.rnn_layers, batch_size, 
                self.rnn_hidden_size, device = self.device)
            cell_encoder = torch.zeros(self.rnn_layers, batch_size, 
                self.rnn_hidden_size, device = self.device)
        # Accomodate batch mismatches
        elif (self.in_train_mode and (hidden_encoder.size()[1] != batch_size or cell_encoder.size()[1] != batch_size)):
            hidden_encoder = torch.zeros(self.rnn_layers, batch_size, 
                self.rnn_hidden_size, device = self.device)
            cell_encoder = torch.zeros(self.rnn_layers, batch_size, 
                self.rnn_hidden_size, device = self.device)

        # If training, feed in true targets to the decoder. Otherwise, feed in
        # predictions.
        if (self.in_train_mode):
            s_err = "Decoder length ({}) does not match target sequence length ({})"
            assert (self.decoder_len == target.size()[1]), s_err.format(self.decoder_len, target.size()[1])
            
            outputs = torch.zeros(batch_size, self.decoder_len, self.output_size, device = self.device)
            mu_collection = torch.zeros(batch_size, self.decoder_len, self.output_size, device = self.device)
            sigma_collection = torch.zeros(batch_size, self.decoder_len, self.output_size, device = self.device)

            # Encoder pass
            self.encoder.flatten_parameters()
            output_encoder, (hidden_encoder, cell_encoder) = \
                self.encoder(input, (hidden_encoder, cell_encoder))
            
            # Decoder pass (single time-step by time-step)
            hidden_decoder = hidden_encoder
            cell_decoder = cell_encoder
            for idx in range(self.decoder_len):
                # Recall that the lstm is element-by-element (sequence size 1)
                decoder_input = torch.unsqueeze(target[:, idx, :], 1) 

                self.decoder.flatten_parameters()
                output_decoder, (hidden_decoder, cell_decoder) = \
                    self.decoder(decoder_input, (hidden_decoder, cell_decoder))
                
                mu = self.squeeze_layers_mu(hidden_decoder.view(self.rnn_hidden_size, 
                    batch_size, self.rnn_layers))
                mu = self.probability_mean(mu.view(batch_size, self.rnn_hidden_size))
                sigma = self.squeeze_layers_sig(hidden_decoder.view(self.rnn_hidden_size, 
                    batch_size, self.rnn_layers))
                sigma = self.probability_std(sigma.view(batch_size, self.rnn_hidden_size))
                sigma = self.std_softplus(sigma)
                distribution = torch.distributions.normal.Normal(mu, sigma)

                mu_collection[:, idx, :] = mu
                sigma_collection[:, idx, :] = sigma
                outputs[:, idx, :] = distribution.sample()
            return outputs, mu_collection, sigma_collection, hidden_encoder, cell_encoder
        else:
            s_err = "Testing with target size {}. Please pass only the most recent observation"
            assert (target.size()[1] == 1), s_err.format(target.size()[1])


            outputs = torch.zeros(batch_size, self.decoder_len, self.output_size, device = self.device)
            mu_collection = torch.zeros(batch_size, self.decoder_len, self.output_size, device = self.device)
            sigma_collection = torch.zeros(batch_size, self.decoder_len, self.output_size, device = self.device)


            # encoder pass
            hidden_encoder = torch.zeros(self.rnn_layers, batch_size, 
                self.rnn_hidden_size, device = self.device)
            cell_encoder = torch.zeros(self.rnn_layers, batch_size, 
                self.rnn_hidden_size, device = self.device)

            self.encoder.flatten_parameters()
            output_encoder, (hidden_encoder, cell_encoder) = self.encoder(input, (hidden_encoder, cell_encoder))


            # Decoder pass (single time-step by time-step)
            hidden_decoder = hidden_encoder
            cell_decoder = cell_encoder
            for idx in range(self.decoder_len):
                decoder_input = target 

                self.decoder.flatten_parameters()
                output_decoder, (hidden_decoder, cell_decoder) = self.decoder(decoder_input, (hidden_decoder, cell_decoder))

                mu = self.squeeze_layers_mu(hidden_decoder.view(self.rnn_hidden_size, 
                    batch_size, self.rnn_layers))
                mu = self.probability_mean(mu.view(batch_size, self.rnn_hidden_size))
                sigma = self.squeeze_layers_sig(hidden_decoder.view(self.rnn_hidden_size, 
                    batch_size, self.rnn_layers))
                sigma = self.probability_std(sigma.view(batch_size, self.rnn_hidden_size))
                sigma = self.std_softplus(sigma)
                distribution = torch.distributions.normal.Normal(mu, sigma)

                mu_collection[:, idx, :] = mu
                sigma_collection[:, idx, :] = sigma
                outputs[:, idx, :] = distribution.sample()
                target = torch.unsqueeze(outputs[:, idx, :], 2).clone().detach().to(device = self.device)
            return outputs, mu_collection, sigma_collection, None, None


def loss_deepAR(mu_collection, sigma_collection, target):
    """
    Calculate the negative log-likelihood over the decoder length.[*] 
    
    [*] - Edit this code if using a non-Gaussian distribution.
    [*] - Edit this code for non-one output size
    -----------------------------------------------------------------------
    :param mu_collection: Tensor of shape [batch_size, decoder_len, 
    output_size]
    :param sigma_collection: Tensor of shape [batch_size, decoder_len,
    output_size]
    :param target: Target output tensor of shape [batch_size, decoder_len, 
        output_size]. 
    """
    distribution = torch.distributions.normal.Normal(mu_collection, sigma_collection)
    likelihood = torch.sum(distribution.log_prob(target))
    return -likelihood
