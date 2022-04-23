import torch
import torch.nn as nn
# from .Model import Model
import numpy
from numpy import fft
import torch.nn.functional as F
from .basic_model import BasicModel
from ...py.util.util import to_var


class HolE(BasicModel):

	def __init__(self, kgs, args, dim = 100, margin = None, epsilon = None):
		super(HolE, self).__init__(args, kgs)

		self.dim = dim
		self.margin = margin
		self.epsilon = epsilon
		self.low_values = False
		self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim)
		self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim)

		if margin == None or epsilon == None:
			nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
			nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
		else:
			self.embedding_range = nn.Parameter(
				torch.Tensor([(self.margin + self.epsilon) / self.dim]), requires_grad=False
			)
			nn.init.uniform_(
				tensor = self.ent_embeddings.weight.data, 
				a = -self.embedding_range.item(), 
				b = self.embedding_range.item()
			)
			nn.init.uniform_(
				tensor = self.rel_embeddings.weight.data, 
				a= -self.embedding_range.item(), 
				b= self.embedding_range.item()
			)
		self.ent_embeddings.weight.data = F.normalize(self.ent_embeddings.weight.data, 2, -1)
		self.rel_embeddings.weight.data = F.normalize(self.rel_embeddings.weight.data, 2, -1)

	def _ccorr(self, r):
		b_size, dim = r.shape
		x = r.view(b_size, 1, dim)
		return torch.cat([x.roll(i, dims=2) for i in range(dim)], dim=1)

	def calc(self, h, t, r):
		r = self._ccorr(r)
		hr = torch.matmul(h.view(-1, 1, self.dim), r)
		return -(hr.view(-1, self.dim) * t).sum(dim=1)

	def forward(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = F.normalize(self.ent_embeddings(batch_h), 2, -1)
		t = F.normalize(self.ent_embeddings(batch_t), 2, -1)
		r = self.rel_embeddings(batch_r)
		score = self.calc(h, t, r)
		return score

	def get_embeddings(self, h_id, r_id, t_id, mode='entities'):
		h_id = to_var(h_id, self.device)
		r_id = to_var(r_id, self.device)
		t_id = to_var(t_id, self.device)
		h_emb = self.ent_embeddings(h_id)
		t_emb = self.ent_embeddings(t_id)
		r_mat = self._ccorr(self.rel_embeddings(r_id))
		b_size = len(h_id)
		if mode == 'entities':
			candidates = self.ent_embeddings.weight.data.view(1, self.ent_tot, self.dim)
			candidates = candidates.expand(b_size, self.ent_tot, self.dim)
		else:
			r_mat = self._ccorr(self.rel_embeddings.weight.data)
			candidates = r_mat.view(1, self.rel_tot, self.dim, self.dim)
			candidates = candidates.expand(b_size, self.rel_tot, self.dim, self.dim)

		return h_emb, t_emb, r_mat, candidates

	def get_score(self, h, r, t):
		b_size = h.shape[0]
		if len(t.shape) == 3:
			assert (len(h.shape) == 2) & (len(r.shape) == 3)
			# this is the tail completion case in link prediction
			h = h.view(b_size, 1, self.dim)
			hr = torch.matmul(h, r).view(b_size, self.dim, 1)
			return -(hr * t.transpose(1, 2)).sum(dim=1)
		elif len(h.shape) == 3:
			assert (len(t.shape) == 2) & (len(r.shape) == 3)
			# this is the head completion case in link prediction
			t = t.view(b_size, self.dim, 1)
			return -(h.transpose(1, 2) * torch.matmul(r, t)).sum(dim=1)
		elif len(r.shape) == 4:
			assert (len(h.shape) == 2) & (len(t.shape) == 2)
			# this is the relation completion case in link prediction
			h = h.view(b_size, 1, 1, self.dim)
			t = t.view(b_size, 1, self.dim)
			hr = torch.matmul(h, r).view(b_size, self.rel_tot, self.dim)
			return -(hr * t).sum(dim=2)

	def regularization(self, data):
		batch_h = data['batch_h']
		batch_t = data['batch_t']
		batch_r = data['batch_r']
		h = self.ent_embeddings(batch_h)
		t = self.ent_embeddings(batch_t)
		r = self.rel_embeddings(batch_r)
		regul = (torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)) / 3
		return regul

	def l3_regularization(self):
		return (self.ent_embeddings.weight.norm(p = 3)**3 + self.rel_embeddings.weight.norm(p = 3)**3)

	def predict(self, data):
		score = -self.forward(data)
		return score.cpu().data.numpy()
