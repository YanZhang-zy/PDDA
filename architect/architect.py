import copy
import torch


class Architect():

    def __init__(self, model, opt):

        self.model = model
        self.v_model = copy.deepcopy(model)
        self.encoder_momentum = opt.momentum
        self.encoder_weight_decay = opt.weight_decay

    def virtual_step(self, input_syn, target_syn, lr, Model_encoder_optim):

        loss = self.model.loss_syn(input_syn, target_syn)[1]

        gradients = torch.autograd.grad(loss, self.model.encoder.parameters())

        with torch.no_grad():

            for w, vw, g in zip(self.model.encoder.parameters(), self.v_model.encoder.parameters(), gradients):
                m = Model_encoder_optim.state[w].get('momentum_buffer', 0.) * self.encoder_momentum
                vw.copy_(w - lr * (m + g + self.encoder_weight_decay * w))

            for a, va in zip(self.model.decoder.parameters(), self.v_model.decoder.parameters()):
                va.copy_(a)

    def unrolled_backward(self, input_syn, target_syn, input_rea, target_rea, lr, Model_encoder_optim):

        self.virtual_step(input_syn, target_syn, lr, Model_encoder_optim)

        loss = self.v_model.loss_rea(input_rea, target_rea)[1]

        v_alphas = tuple(self.v_model.decoder.parameters())
        v_weights = tuple(self.v_model.encoder.parameters())
        v_grads = torch.autograd.grad(loss, v_alphas + v_weights)
        dalpha = v_grads[:len(v_alphas)]
        dw = v_grads[len(v_alphas):]

        hessian = self.compute_hessian(dw, input_syn, target_syn)

        with torch.no_grad():
            for alpha, da, h in zip(self.model.decoder.parameters(), dalpha, hessian):
                alpha.grad = da - lr * h

    def compute_hessian(self, dw, input_syn, target_syn):
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        with torch.no_grad():
            for p, d in zip(self.model.encoder.parameters(), dw):
                p += eps * d
        loss = self.model.loss_syn(input_syn, target_syn)[1]
        dalpha_pos = torch.autograd.grad(loss, self.model.decoder.parameters())

        with torch.no_grad():
            for p, d in zip(self.model.encoder.parameters(), dw):
                p -= 2. * eps * d
        loss = self.model.loss_syn(input_syn, target_syn)[1]
        dalpha_neg = torch.autograd.grad(loss, self.model.decoder.parameters())

        with torch.no_grad():
            for p, d in zip(self.model.encoder.parameters(), dw):
                p += eps * d

        hessian = [(p - n) / 2. * eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
