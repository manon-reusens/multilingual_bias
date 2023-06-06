from tqdm import tqdm 
import torch

class DensRay:
    def __init__(self, Lemb, Remb):
        self.lemb = Lemb
        self.remb = Remb

    def fit(self, weights=None, normalize_D=True):
        """Fit DensRay
        Args:
            weights: only for binary model; how to weight the two
                summands; if none
                
                
                : apply dynamic weighting. Example input: [1.0, 1.0]
            normalize_D: bool whether to normalize the difference vectors with l2 norm
        """
        #self.computeA_binary_part1(normalize_D=normalize_D)
        print(type(self.lemb))
        self.A_equal = self.opsum(self.lemb) + self.opsum(self.remb)
        self.A_unequal = self.opsum(self.lemb, self.remb) + self.opsum(self.remb, self.lemb)
        self.computeA_binary_part2(weights=weights)
        self.compute_trafo()
        self.compute_mean_var()

    @staticmethod
    def opsum(a, b=None):
        if b is None: b = a
        out = -torch.ger(a.sum(dim=0), b.sum(dim=0))
        out = out + out.T
        out += b.shape[0] * torch.mm(a.T,a)
        out += a.shape[0] * torch.mm(b.T,b)
        return out

    @staticmethod
    def outer_product_sub_binary(v, M, normD):
        """Helper function to compute the sum of outer products

        While it is not very readable, it is more efficient than
        a brute force implementation.
        """
        d = v.unsqueeze(0) - M
        if normD:
            norm = d.norm(dim=1)
            norm[norm == 0] = 1
            d = d / (norm.unsqueeze(0).T)
        return torch.mm(d.T, d)
    
    def computeA_binary_part1(self, normalize_D=False):
        """First part of computing the matrix A.
        Args:
            normalize_D: bool whether to normalize the difference vectors with l2 norm.
        """
        dim = self.lemb.shape[1]
        self.A_equal = torch.zeros((dim, dim)).to(device)
        self.A_unequal = torch.zeros((dim, dim)).to(device)
        for ipos in tqdm.trange(self.lemb.shape[0]):
            v = self.lemb[ipos]
            self.A_equal += self.outer_product_sub_binary(v, self.lemb, normalize_D)
            self.A_unequal += self.outer_product_sub_binary(v, self.remb, normalize_D)
        for ineg in tqdm.trange(self.remb.shape[0]):
            v = self.remb[ineg]
            self.A_equal += self.outer_product_sub_binary(v, self.remb, normalize_D)
            self.A_unequal += self.outer_product_sub_binary(v, self.lemb, normalize_D)

    def computeA_binary_part2(self, weights=None):
        """Second part of computing the matrix A.
        Args:
            weights: only for binary model; how to weight the two 
                summands; if none: apply dynamic weighting. Example input: [1.0, 1.0]
        """
        if weights is None:
            weights = [1 / (2 * self.lemb.shape[0] * self.remb.shape[0]), 1 /
                       (self.lemb.shape[0]**2 + self.remb.shape[0]**2)]
        # normalize matrices for numerical reasons
        # note that this does not change the eigenvectors
        n1 = self.A_unequal.max()
        n2 = self.A_equal.max()
        weights = [weights[0] / max(n1, n2), weights[1] / max(n1, n2)]
        self.A = weights[0] * self.A_unequal - weights[1] * self.A_equal

    def compute_trafo(self):
        """Given A, this function computes the actual Transformation.
        It essentially just does an eigenvector decomposition.
        """
        eigvals, eigvecs = self.A.symeig(eigenvectors=True)
        # need to sort the eigenvalues
        idx = eigvals.argsort(descending=True)
        eigvals, self.eigvecs = eigvals[idx], eigvecs[:, idx]
    
    def compute_mean_var(self):
        first_dim = torch.mm(torch.cat((self.lemb, self.remb)), self.eigvecs)[:, 0]
        self.mean = first_dim.mean()
        self.std = first_dim.var().sqrt()
