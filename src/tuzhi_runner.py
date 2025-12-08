# TODO:1.初始化dbde网络
from DBDE_diffusion import DBDEDiffusion
"""new implementation"""
class ObsEncodeNet(nn.Module):
	#观测编码网络
	"""
	input delayed state and action, output the non-delayed state
	"""
	def __init__(
		self,
		state_shape: Sequence[int],
		action_shape: Sequence[int],
		global_cfg: Dict[str, Any],
		**kwargs,
	) -> None:
		super().__init__()
		self.global_cfg = global_cfg
		self.hps = kwargs

		# cal input dim，输入维度
		selected_encoder_net = self.hps["encoder_net_mlp"]
		if self.global_cfg.actor_input.history_merge_method in ["cat_mlp"]:
			self.normal_encode_dim = state_shape[0] + action_shape[0] * global_cfg.history_num#合并维度
		elif self.global_cfg.actor_input.history_merge_method in ["stack_rnn"]:
			selected_encoder_net = self.hps["encoder_net_rnn"]
			if self.global_cfg.history_num > 0:
				self.normal_encode_dim = state_shape[0] + action_shape[0]
			else:
				self.normal_encode_dim = state_shape[0]
		elif self.global_cfg.actor_input.history_merge_method in ["none"]:
			self.normal_encode_dim = state_shape[0]
		else:
			raise ValueError("unknown history_merge_method: {}".format(self.global_cfg.actor_input.history_merge_method))
		
		# cal output dim，输出维度
		self.oracle_encode_dim = state_shape[0]

		self.feat_dim = self.hps["feat_dim"]
		self.normal_encoder_net = selected_encoder_net(self.normal_encode_dim, self.feat_dim, device=self.hps["device"], head_num=2)
		self.oracle_encoder_net = selected_encoder_net(self.oracle_encode_dim, self.feat_dim, device=self.hps["device"], head_num=2)
		self.decoder_net = self.hps["decoder_net"](self.feat_dim, self.oracle_encode_dim, device=self.hps["device"], head_num=1)
		self.normal_encoder_net.to(self.hps["device"])
		self.oracle_encoder_net.to(self.hps["device"])
		self.decoder_net.to(self.hps["device"])
		
	def forward(
		self,
		input: Union[np.ndarray, torch.Tensor],
		info: Dict[str, Any] = {},
	) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
		raise ValueError("should call normal_encode or oracle_encode")

	def normal_encode(
			self,
			input: Union[np.ndarray, torch.Tensor],
			info: Dict[str, Any] = {},
		):
		return self.net_forward(self.normal_encoder_net, input, info)
	
	def oracle_encode(
			self,
			input: Union[np.ndarray, torch.Tensor],
			info: Dict[str, Any] = {},
		):
		return self.net_forward(self.oracle_encoder_net, input, info)
	
	def net_forward(self, net, input, info):
		info = {}
		(mu, logvar), state_ = net(input)
		# feats = self.vae_sampling(mu, logvar)
		feats = self.torch_sampling(mu, logvar)
		info["mu"] = mu
		info["logvar"] = logvar
		info["state"] = state_
		info["feats"] = feats
		return feats, info

	def vae_sampling(self, mu, log_var):
		std = torch.exp(0.5*log_var)
		eps = torch.randn_like(std)
		return eps.mul(std).add_(mu)
	
	def torch_sampling(self, mu, log_var):
		z_dist = Normal(mu, torch.exp(0.5*log_var))
		z = z_dist.rsample()
		return z

	def decode(
			self,
			input: Union[np.ndarray, torch.Tensor],
			info: Dict[str, Any] = {},
		):
		"""
		ps. there is only one type of decoder since it is always from latent dim to the oracle obs
		"""
		info = {}
		encoder_outputs, state_ = self.decoder_net(input)
		res = encoder_outputs[0]
		return res, info

class DefaultRLRunner:
	""" Runner for RL algorithms
	Work flow: 
		entry.py
			initialize cfg.runner, pass cfg to runner
		runner.py(XXXRunner)
			initialize envs, policy, buffer, etc.
			call DefaultRLRunner.__init__(cfg) for mmon initialization
			call XXXRunner.__init__(cfg) for specific initialization
	"""
	def start(self, cfg):
		self.cfg = cfg
		self.env = cfg.env
		# init
		cfg.env.global_cfg = self.cfg.global_cfg
		seed = int(time()) if cfg.seed is None else cfg.seed
		utils.seed_everything(seed) # TODO add env seed
		self.env = utils.make_env(cfg.env) # to get obs_space and act_space
		self.console = Console()
		self.log = self.console.log
		self.log("Logger init done!")
		self.log("Checking cfg ...")
		self.check_cfg()
		self.log("_init_basic_components_for_all_alg ...")
		self._init_basic_components_for_all_alg()

	def _init_basic_components_for_all_alg(self):
		cfg = self.cfg
		env = self.env
		# basic for all algorithms
		self.global_cfg = cfg.global_cfg
		self.log("Init buffer & collector ...")
		self.buf = cfg.buffer
		self.train_collector = EnvCollector(env)
		self.test_collector = EnvCollector(env)
		self.log("Init others ...")
		self.record = WaybabaRecorder()
		self._start_time = time()
		if self.cfg.trainer.progress_bar:
			self.progress = Progress()
			self.progress.start()

	def check_cfg(self):
		print("You should implement check_cfg in your runner!")

class OfflineRLRunner(DefaultRLRunner):
	def start(self, cfg):
		super().start(cfg)
		self.log("init_components ...")
		#初始化网络结构
		self.init_components()

		self.log("_initial_exploration ...")
		self._initial_exploration()

		self.log("Training Start ...")
		self.env_step_global = 0
		if cfg.trainer.progress_bar: self.training_task = self.progress.add_task("[green]Training...", total=cfg.trainer.max_epoch*cfg.trainer.step_per_epoch)
		
		while True: # traininng loop
			# env step collect
			self._collect_once()
			
			# update
			if self._should_update(): 
				for _ in range(int(cfg.trainer.step_per_collect/cfg.trainer.update_per_step)):
					self.update_once()
			
			# evaluate
			if self._should_evaluate():
				self._evaluate()
			
			#log
			if self._should_write_log():
				self._log_time()
				self.record.upload_to_wandb(step=self.env_step_global, commit=False)
			
			# upload
			if self._should_upload_log():
				wandb.log({}, commit=True)
				
			# loop control
			if self._should_end(): break
			if cfg.trainer.progress_bar: self.progress.update(self.training_task, advance=cfg.trainer.step_per_collect, description=f"[green]🚀 Training {self.env_step_global}/{self.cfg.trainer.max_epoch*self.cfg.trainer.step_per_epoch}[/green]\n"+self.record.to_progress_bar_description())
			self.env_step_global += self.cfg.trainer.step_per_collect

		self._end_all()

	def init_components(self):
		raise NotImplementedError

	def _initial_exploration(self):
		"""exploration before training and add to self.buf"""
		initial_batches, info_ = self.train_collector.collect(
			act_func="random", n_step=self.cfg.start_timesteps, reset=True, 
			progress_bar="Initial Exploration ..." if self.cfg.trainer.progress_bar else None,
			rich_progress=self.progress if self.cfg.trainer.progress_bar else None
		)
		self.train_collector.reset()
		for batch in initial_batches: self.buf.add(batch)

	def _collect_once(self):
		"""collect data and add to self.buf"""

		batches, info_ = self.train_collector.collect(
			act_func=partial(self.select_act_for_env, mode="train"), 
			n_step=self.cfg.trainer.step_per_collect, reset=False
		)
		for batch in batches: self.buf.add(batch)

		# store history
		if not hasattr(self, "rew_sum_history"): self.rew_sum_history = []
		if not hasattr(self, "ep_len_history"): self.ep_len_history = []
		self.rew_sum_history += info_["rew_sum_list"]
		self.ep_len_history += info_["ep_len_list"]
		res_info = {
			"batches": batches,
			**info_
		}
		self._on_collect_end(**res_info)

	def _on_collect_end(self, **kwargs):
		"""called after a step of data collection"""
		self.on_collect_end(**kwargs)
	
	def update_once(self):
		raise NotImplementedError

	def _evaluate(self):
		"""Evaluate the performance of an agent in an environment.
		Args:
			env: Environment to evaluate on.
			act_func: Action selection function. It should take a single argument
				(observation) and return a single action.
		Returns:
			Episode reward.
		"""
		if not hasattr(self, "epoch_cnt"): self.epoch_cnt = 0
		# evaluate
		for mode in ["eval", "train"]:
			eval_type = "deterministic" if mode == "eval" else ""
			eval_batches, _ = self.test_collector.collect(
				act_func=partial(self.select_act_for_env, mode=mode), 
				n_episode=self.cfg.trainer.episode_per_test, reset=True,
				progress_bar=f"Evaluating {eval_type} ..." if self.cfg.trainer.progress_bar else None,
				rich_progress=self.progress if self.cfg.trainer.progress_bar else None,
			)
			eval_rews = [0. for _ in range(self.cfg.trainer.episode_per_test)]
			eval_lens = [0 for _ in range(self.cfg.trainer.episode_per_test)]
			cur_ep = 0
			for i, batch in enumerate(eval_batches): 
				eval_rews[cur_ep] += batch.rew
				eval_lens[cur_ep] += 1
				if batch.terminated or batch.truncated:
					cur_ep += 1
			self.record("eval/rew_mean"+"_"+eval_type, np.mean(eval_rews))
			self.record("eval/len_mean"+"_"+eval_type, np.mean(eval_lens))
		# loop control
		self.epoch_cnt += 1
		self.record("epoch", self.epoch_cnt)
		self._on_evaluate_end()
	
	def _on_evaluate_end(self):
		# print epoch log (deactivated in debug mode as the info is already in progress bar)
		if not self.cfg.trainer.hide_eval_info_print: 
			to_print = self.record.__str__().replace("\n", "  ")
			to_print = "[Epoch {: 5d}/{}] ### ".format(self.epoch_cnt-1, self.cfg.trainer.max_epoch) + to_print
			print(to_print)
		self.on_evaluate_end()
	
	def on_evaluate_end(self):
		pass

	def _log_time(self):
		if self.env_step_global > 1000:
			cur_time = time()
			hours_spent = (cur_time-self._start_time) / 3600
			speed = self.env_step_global / hours_spent
			hours_left = (self.cfg.trainer.max_epoch*self.cfg.trainer.step_per_epoch-self.env_step_global) / speed
			self.record("misc/hours_spent", hours_spent)
			self.record("misc/hours_left", hours_left)
			self.record("misc/step_per_hour", speed)

	def _end_all(self):
		if self.cfg.trainer.progress_bar: self.progress.stop()
		if self.cfg.env.save_minari: # save dataset
			version_ = self.cfg.trainer.max_epoch * self.cfg.trainer.step_per_epoch
			dataset_id = self.cfg.env.name.lower().split("-")[0]+f"-sac_{version_}"+"-v0"
			self.env.create_dataset(dataset_id=dataset_id)
			print("Minari dataset saved as name {}".format(dataset_id))
			# mv ~/.minari/datasets/{dataset_id} to os.environ['UDATADIR'] + /minari/datasets/{dataset_id}
			# if exist, replace
			import shutil
			source_dir = os.path.join(os.path.expanduser("~"), ".minari", "datasets", dataset_id)
			dest_dir = os.path.join(os.environ['UDATADIR'], "minari", "datasets")
			if os.path.exists(dest_dir+"/"+dataset_id): shutil.rmtree(dest_dir+"/"+dataset_id)
			if not os.path.exists(dest_dir): os.makedirs(dest_dir)
			shutil.move(source_dir, dest_dir)
			print("Dataset moved from {} to {}".format(source_dir, dest_dir))

	def select_act_for_env(self, batch, state, mode):
		"""
		Note this is only used when interacting with env. For learning state,
		the actions are got by calling self.actor ...
		Usage: 
			would be passed as a function to collector
		Args:
			batch: batch of data
			state: {"hidden": [], "hidden_pred": [], ...}
			mode: "train" or "eval"
				for train, it would use stochastic action
				for eval, it would use deterministic action
				usage: 
					1. when collecting data, mode="train" is used
					2. when evaluating, both mode="train" and mode="eval" are used
		"""
		raise NotImplementedError

	def _should_update(self):
		# TODO since we collect x steps, so we always update
		# if not hasattr(self, "should_update_record"): self.should_update_record = {}
		# cur_update_tick = self.env_step_global // self.cfg.trainer.
		# if cur_update_tick not in self.should_update_record:
		# 	self.should_update_record[cur_update_tick] = True
		# 	return True
		# return False
		return True
	
	def _should_evaluate(self):
		if not hasattr(self, "should_evaluate_record"): self.should_evaluate_record = {}
		cur_evaluate_tick = self.env_step_global // self.cfg.trainer.step_per_epoch
		if cur_evaluate_tick not in self.should_evaluate_record:
			self.should_evaluate_record[cur_evaluate_tick] = True
			return True
		return False
	
	def _should_write_log(self):
		if not hasattr(self, "should_log_record"): self.should_log_record = {}
		cur_log_tick = self.env_step_global // self.cfg.trainer.log_interval
		if cur_log_tick not in self.should_log_record:
			self.should_log_record[cur_log_tick] = True
			return True
		return False

	def _should_upload_log(self):
		if not self.cfg.trainer.log_upload_interval: return True # missing or zero, always upload
		if not hasattr(self, "should_upload_record"): self.should_upload_record = {}
		cur_upload_tick = self.env_step_global // self.cfg.trainer.log_upload_interval
		if cur_upload_tick not in self.should_upload_record:
			self.should_upload_record[cur_upload_tick] = True
			return True
		return False

	def _should_end(self):
		return self.env_step_global >= self.cfg.trainer.max_epoch * self.cfg.trainer.step_per_epoch

class TD3SACRunner(OfflineRLRunner):
	def check_cfg(self):
		cfg = self.cfg
		global_cfg = cfg.global_cfg
		if global_cfg.critic_input.history_merge_method == "stack_rnn":
			pass

	def init_components(self):
		self.log("Init networks ...")
		env = self.env
		cfg = self.cfg

		# networks
		self.actor = cfg.actor(state_shape=env.observation_space.shape, action_shape=env.action_space.shape, max_action=env.action_space.high[0],global_cfg=self.cfg.global_cfg).to(cfg.device)
		self.actor_optim = cfg.actor_optim(self.actor.parameters())
		self.actor_old = deepcopy(self.actor)
		
		# decide bi or two direction
		if cfg.global_cfg.critic_input.history_merge_method == "stack_rnn":
			if cfg.global_cfg.critic_input.bi_or_si_rnn == "si":
				self.critic1 = cfg.critic1(env.observation_space.shape, action_shape=env.action_space.shape, bidirectional=False, global_cfg=self.cfg.global_cfg).to(cfg.device)
				self.critic2 = cfg.critic2(env.observation_space.shape, action_shape=env.action_space.shape, bidirectional=False, global_cfg=self.cfg.global_cfg).to(cfg.device)
			elif cfg.global_cfg.critic_input.bi_or_si_rnn == "bi":
				self.critic1 = cfg.critic1(env.observation_space.shape, action_shape=env.action_space.shape, bidirectional=True, global_cfg=self.cfg.global_cfg).to(cfg.device)
				self.critic2 = cfg.critic2(env.observation_space.shape, action_shape=env.action_space.shape, bidirectional=True, global_cfg=self.cfg.global_cfg).to(cfg.device)
			elif cfg.global_cfg.critic_input.bi_or_si_rnn == "both":
				self.critic1 = cfg.critic1(env.observation_space.shape, action_shape=env.action_space.shape, bidirectional=True, global_cfg=self.cfg.global_cfg).to(cfg.device)
				self.critic2 = cfg.critic2(env.observation_space.shape, action_shape=env.action_space.shape, bidirectional=True, global_cfg=self.cfg.global_cfg).to(cfg.device)
				self.critic_sirnn_1 = cfg.critic1(env.observation_space.shape, action_shape=env.action_space.shape, bidirectional=False, global_cfg=self.cfg.global_cfg).to(cfg.device)
				self.critic_sirnn_2 = cfg.critic2(env.observation_space.shape, action_shape=env.action_space.shape, bidirectional=False, global_cfg=self.cfg.global_cfg).to(cfg.device)
				self.critic_sirnn_1_optim = cfg.critic1_optim(self.critic_sirnn_1.parameters())
				self.critic_sirnn_2_optim = cfg.critic2_optim(self.critic_sirnn_2.parameters())
			else:
				raise NotImplementedError
		else:
			self.critic1 = cfg.critic1(env.observation_space.shape, action_shape=env.action_space.shape, bidirectional=False, global_cfg=self.cfg.global_cfg).to(cfg.device)
			self.critic2 = cfg.critic2(env.observation_space.shape, action_shape=env.action_space.shape, bidirectional=False, global_cfg=self.cfg.global_cfg).to(cfg.device)
		self.critic1_optim = cfg.critic1_optim(self.critic1.parameters())
		self.critic2_optim = cfg.critic2_optim(self.critic2.parameters())
		self.critic1_old = deepcopy(self.critic1)
		self.critic2_old = deepcopy(self.critic2)

		self.actor_old.eval()
		self.critic1.train()
		self.critic2.train()
		self.critic1_old.train()
		self.critic2_old.train()
		
		if self.ALGORITHM == "q_sac":
			redudent_net = ["actor_old"]
			self.critic1_old.eval()
			self.critic2_old.eval()
			self.log("init sac alpha ...")
			self._init_sac_alpha()
		if self.ALGORITHM == "sac":
			redudent_net = ["actor_old"]
			self.critic1_old.eval()
			self.critic2_old.eval()
			self.log("init sac alpha ...")
			self._init_sac_alpha()

		
		if redudent_net:
			for net_name in redudent_net: delattr(self, net_name)
		# obs pred & encode
		assert not (self.global_cfg.actor_input.obs_pred.turn_on and self.global_cfg.actor_input.obs_encode.turn_on), "obs_pred and obs_encode cannot be used at the same time"
		

		if self.global_cfg.actor_input.obs_pred.turn_on:
			self.pred_net = self.global_cfg.actor_input.obs_pred.net(
				state_shape=self.env.observation_space.shape,
				action_shape=self.env.action_space.shape,
				global_cfg=self.global_cfg,
			)
			self._pred_optim = self.global_cfg.actor_input.obs_pred.optim(
				self.pred_net.parameters(),
			)
			if self.global_cfg.actor_input.obs_pred.auto_kl_target:
				self.kl_weight_log = torch.tensor([np.log(
					self.global_cfg.actor_input.obs_pred.norm_kl_loss_weight
				)], device=self.actor.device, requires_grad=True)
				self._auto_kl_optim = self.global_cfg.actor_input.obs_pred.auto_kl_optim([self.kl_weight_log])
			else:
				self.kl_weight_log = torch.tensor([np.log(
					self.global_cfg.actor_input.obs_pred.norm_kl_loss_weight
				)], device=self.actor.device)
		# TODO :写一下diffusion model的初始化方法
		if self.ALGORITHM == "q_sac":
			state_dim=self.env.observation_space.shape[0]
			cond_dim=self.env.observation_space.shape[0] + self.env.action_space.shape[0] * self.global_cfg.history_num
			self.dbde = DBDEDiffusion(
					state_dim=state_dim,
					cond_dim=cond_dim,
					num_steps=self.global_cfg.actor_input.obs_pred.num_steps,
					hidden_dim=self.global_cfg.actor_input.obs_pred.feat_dim,
					device=self.cfg.device,
				)
			self.dbde_optimizer = torch.optim.Adam(self.dbde.parameters(), lr=1e-3)
		if self.global_cfg.actor_input.obs_encode.turn_on:
			self.encode_net = self.global_cfg.actor_input.obs_encode.net(
				state_shape=self.env.observation_space.shape,
				action_shape=self.env.action_space.shape,
				global_cfg=self.global_cfg,
			)
			self._encode_optim = self.global_cfg.actor_input.obs_encode.optim(
				self.encode_net.parameters(),
			)#创建优化器
			if self.global_cfg.actor_input.obs_encode.auto_kl_target:
				self.kl_weight_log = torch.tensor([np.log(
					self.global_cfg.actor_input.obs_encode.norm_kl_loss_weight
				)], device=self.actor.device, requires_grad=True)
				self._auto_kl_optim = self.global_cfg.actor_input.obs_encode.auto_kl_optim([self.kl_weight_log])
			else:
				self.kl_weight_log = torch.tensor([np.log(
					self.global_cfg.actor_input.obs_encode.norm_kl_loss_weight
				)], device=self.actor.device)
	
	def preprocess_from_env(self, batch, state, mode=None):
		#这里主要还是处理actor的输入
		"""
		use in the online interaction with env
		"""
		assert len(batch.obs.shape) == 1, "for online batch, batch size must be 1"
		res_state = {}

		# add "hidden" since it is not used in this function
		res_state = update_state(res_state, state, {"hidden": "hidden"})
		#normal是延迟的观测，
		if self.global_cfg.actor_input.obs_type == "normal": a_in = batch.obs
		elif self.global_cfg.actor_input.obs_type == "oracle": a_in = batch.info["obs_next_nodelay"]

		if self.global_cfg.actor_input.history_merge_method == "none":
			if self.global_cfg.actor_input.obs_pred.turn_on:
				pred_out, pred_info = self.pred_net(a_in)
				if self.global_cfg.actor_input.obs_pred.input_type == "obs":
					a_in = pred_out.cpu()
				elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
					a_in = pred_info["feats"].cpu()
				else: raise ValueError("unknown input_type: {}".format(self.global_cfg.actor_input.obs_pred.input_type))
				pred_abs_error_online = ((pred_out - torch.tensor(batch.info["obs_next_nodelay"],device=self.cfg.device))**2).mean().item()
				self.record("obs_pred/pred_abs_error_online", pred_abs_error_online)
			elif self.global_cfg.actor_input.obs_encode.turn_on:
				encode_output, encode_info = self.encode_net.normal_encode(a_in)
				a_in = encode_output.cpu()
				if self.global_cfg.actor_input.obs_encode.pred_loss_weight:
					pred_obs_output_cur, _ = self.encode_net.decode(encode_output)
					pred_abs_error_online = ((pred_obs_output_cur - torch.tensor(batch.info["obs_next_nodelay"],device=self.cfg.device))**2).mean().item()
					self.record("obs_pred/pred_abs_error_online", pred_abs_error_online)
		
		elif self.global_cfg.actor_input.history_merge_method == "cat_mlp":
			if self.global_cfg.history_num > 0: # only cat when > 0
				a_in = np.concatenate([
					a_in,
					batch.info["historical_act"].flatten() if not self.cfg.global_cfg.debug.new_his_act \
					else batch.info["historical_act_next"].flatten()
			], axis=-1)
			#history数据融合
			if self.global_cfg.actor_input.obs_pred.turn_on:
				pred_out, pred_info = self.pred_net(a_in)
				if self.global_cfg.actor_input.obs_pred.input_type == "obs":
					a_in = pred_out.cpu()
				elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
					a_in = pred_info["feats"].cpu()
				else: raise ValueError("unknown input_type: {}".format(self.global_cfg.actor_input.obs_pred.input_type))
				#预测损失，
				pred_abs_error_online = ((pred_out - torch.tensor(batch.info["obs_next_nodelay"],device=self.cfg.device))**2).mean().item()
				self.record("obs_pred/pred_abs_error_online", pred_abs_error_online)
			
			elif self.global_cfg.actor_input.obs_encode.turn_on:
				encode_output, encode_info = self.encode_net.normal_encode(a_in)
				a_in = encode_output.cpu()
				#这里是否需要编码预测损失
				if self.global_cfg.actor_input.obs_encode.pred_loss_weight:
					pred_obs_output_cur, _ = self.encode_net.decode(encode_output)
					pred_abs_error_online = ((pred_obs_output_cur - torch.tensor(batch.info["obs_next_nodelay"],device=self.cfg.device))**2).mean().item()
					self.record("obs_pred/pred_abs_error_online", pred_abs_error_online)
		
		else:
			raise ValueError(f"history_merge_method {self.global_cfg.actor_input.history_merge_method} not implemented")
		
		return a_in, res_state
	
	def select_act_for_env(self, batch, state, mode):
		a_in, res_state = self.preprocess_from_env(batch, state, mode=mode)
		
		# forward
		if not isinstance(a_in, torch.Tensor): a_in = torch.tensor(a_in, dtype=torch.float32).to(self.cfg.device)
		if self.cfg.global_cfg.debug.abort_infer_state:
			state_for_a = None
		else:
			state_for_a = distill_state(state, {"hidden": "hidden"})
		a_out, actor_state = self.actor(a_in, state_for_a)
		res_state = update_state(res_state, actor_state, {"hidden": "hidden"})
		
		if self.ALGORITHM == "td3":
			if mode == "train":
				a_out = a_out[0]
				# noise = torch.tensor(self._noise(a_out.shape), device=self.cfg.device)
				a_scale = torch.tensor((self.env.action_space.high - self.env.action_space.low) / 2.0, dtype=torch.float32)
				noise = torch.normal(0., a_scale * self.exploration_noise).to(device=self.cfg.device)
				res = a_out + noise
				# if self.cfg.policy.noise_clip > 0.0:
				# 	noise = noise.clamp(-self.cfg.policy.noise_clip, self.cfg.policy.noise_clip)
				res = res.clip(
					torch.tensor(self.env.action_space.low, device=self.cfg.device),
					torch.tensor(self.env.action_space.high, device=self.cfg.device),
				)
			elif mode == "eval":
				res = a_out[0]
			else: 
				raise ValueError("unknown mode: {}".format(mode))
		elif self.ALGORITHM == "sac":
			if mode == "train":
				assert isinstance(a_out, tuple) # (mean, logvar)
				dist = Independent(Normal(*a_out), 1)
				act = dist.rsample()
				squashed_action = torch.tanh(act) * torch.tensor(self.env.action_space.high, device=self.cfg.device) + 0.0 # TODO bias
				# log_prob = dist.log_prob(act).unsqueeze(-1)
				# log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) +
				# 								np.finfo(np.float32).eps.item()).sum(-1, keepdim=True) # TODO remove, seems not used 
				res = squashed_action#actor选取的动作
			elif mode == "eval":
				res = a_out[0]
				res = torch.tanh(res) * torch.tensor(self.env.action_space.high, device=self.cfg.device) + 0.0 # TODO bias
			else: raise ValueError("unknown mode: {}".format(mode))
		elif self.ALGORITHM == "q_sac":
			if mode == "train":
				assert isinstance(a_out, tuple) # (mean, logvar)
				dist = Independent(Normal(*a_out), 1)
				act = dist.rsample()
				squashed_action = torch.tanh(act) * torch.tensor(self.env.action_space.high, device=self.cfg.device) + 0.0 # TODO bias
				# log_prob = dist.log_prob(act).unsqueeze(-1)
				# log_prob = log_prob - torch.log((1 - squashed_action.pow(2)) +
				# 								np.finfo(np.float32).eps.item()).sum(-1, keepdim=True) # TODO remove, seems not used 
				res = squashed_action
			elif mode == "eval":
				res = a_out[0]
				res = torch.tanh(res) * torch.tensor(self.env.action_space.high, device=self.cfg.device) + 0.0 # TODO bias
			else: raise ValueError("unknown mode: {}".format(mode))
		elif self.ALGORITHM == "ddpg":
			if mode == "train":
				a_out = a_out[0]
				# noise = torch.tensor(self._noise(a_out.shape), device=self.cfg.device)
				a_scale = torch.tensor((self.env.action_space.high - self.env.action_space.low) / 2.0, dtype=torch.float32)
				noise = torch.normal(0., a_scale * self.exploration_noise).to(device=self.cfg.device)
				res = a_out + noise
				# if self.cfg.policy.noise_clip > 0.0:
				# 	noise = noise.clamp(-self.cfg.policy.noise_clip, self.cfg.policy.noise_clip)
				res = res.clip(
					torch.tensor(self.env.action_space.low, device=self.cfg.device),
					torch.tensor(self.env.action_space.high, device=self.cfg.device),
				)
			elif mode == "eval":
				res = a_out[0]
			else: 
				raise ValueError("unknown mode: {}".format(mode))

		return res, res_state if res_state else None
	
	def on_collect_end(self, **kwargs):
		"""called after a step of data collection"""
		if "rew_sum_list" in kwargs and kwargs["rew_sum_list"]:
			for i in range(len(kwargs["rew_sum_list"])): self.record("collect/rew_sum", kwargs["rew_sum_list"][i])
		if "ep_len_list" in kwargs and kwargs["ep_len_list"]:
			for i in range(len(kwargs["ep_len_list"])): self.record("collect/ep_len", kwargs["ep_len_list"][i])

	def update_once(self):
		#更新actor-critic的函数
		# indices = self.buf.sample_indices(self.cfg.trainer.batch_size)
		idxes, idxes_remaster, valid_mask = self.buf.sample_indices_remaster(self.cfg.trainer.batch_size, self.cfg.trainer.batch_seq_len)
		batch = self._indices_to_batch(idxes)
		batch.valid_mask = valid_mask
		if self._burnin_num(): # TODO move all of this to buffer would make the logic nicer
			burnin_idxes, _, burnin_mask = self.buf.create_burnin_pair(idxes_remaster, self._burnin_num())
			burnin_batch = self._indices_to_batch(burnin_idxes)
			burnin_batch.valid_mask = burnin_mask
			batch = self._pre_update_process(batch, burnin_batch) # would become state in batch.state
		else:
			batch = self._pre_update_process(batch)

		if not hasattr(self, "critic_update_cnt"): self.update_cnt = 0
		
		if self.ALGORITHM == "td3":
			# update cirtic
			self.update_critic(batch)
			if self.update_cnt % self.cfg.policy.update_a_per_c == 0:
				# update actor
				self.update_actor(batch)
				self.exploration_noise *= self.cfg.policy.noise_decay_rate
			self._soft_update(self.actor_old, self.actor, self.cfg.policy.tau)
			self._soft_update(self.critic1_old, self.critic1, self.cfg.policy.tau)
			self._soft_update(self.critic2_old, self.critic2, self.cfg.policy.tau)
		elif self.ALGORITHM == "sac":
			self.update_critic(batch)
			self.update_actor(batch)
			self._soft_update(self.critic1_old, self.critic1, self.cfg.policy.tau)
			self._soft_update(self.critic2_old, self.critic2, self.cfg.policy.tau)
		elif self.ALGORITHM == "ddpg":
			self.update_critic(batch)
			self.update_actor(batch)
			self._soft_update(self.critic1_old, self.critic1, self.cfg.policy.tau)
			self._soft_update(self.actor_old, self.actor, self.cfg.policy.tau)
		elif self.ALGORITHM == "q_sac":
			
			self.update_dbde(batch)
			self.update_critic(batch)
			self.update_actor(batch)
			self._soft_update(self.critic1_old, self.critic1, self.cfg.policy.tau)
			self._soft_update(self.critic2_old, self.critic2, self.cfg.policy.tau)
		else:
			raise NotImplementedError
			
		self.update_cnt += 1
	
	def update_critic(self, batch):
		raise NotImplementedError
	
	def update_actor(self, batch):
		raise NotImplementedError

	def _indices_to_batch(self, indices):
		#从batch里面拿去训练所需要的数据
		""" sample batch from buffer with indices
		After sampling, indices are discarded. So we need to make sure that
		all the information we need is in the batch.

		TODO souce of historical act, now is from info["histo...], can also from act

		Exaplanation of the batch keys:
			` basic ones from the buffer: obs, act, obs_next, rew, done, terminated, truncated, policy`
			dobs, dobs_next (*, obs_dim): delayed obs, delayed obs_next, inherited from obs, obs_next. To
				avoid confusion, the obs, obs_next would be renamed to these two names.
			oobs, oobs_next (*, obs_dim): oracle obs, oracle obs_next
			act, rew, done (*, ): no changes (terminated and truncated are removed since we don't need them)
			ahis_cur, ahis_next (*, history_len, act_dim): history of actions
		"""
		keeped_keys = ["dobs", "dobs_next", "oobs", "oobs_next", "ahis_cur", "ahis_next", "ohis_cur","ohis_next", "act", "rew", "done", "terminated", "obs_delayed_step_num"]
		batch = self.buf[indices]
		batch.dobs, batch.dobs_next = batch.obs, batch.obs_next
		batch.oobs, batch.oobs_next = batch.info["obs_nodelay"], batch.info["obs_next_nodelay"]
		#历史动作集合延迟历史动作的集合
		if self.cfg.global_cfg.debug.new_his_act:
			batch.ahis_cur = batch.info["historical_act_cur"]
			batch.ahis_next = batch.info["historical_act_next"]
			batch.ohis_cur = batch.info["historical_obs_cur"]
			batch.ohis_next = batch.info["historical_obs_next"]
		else:
			batch.ahis_cur = batch.info["historical_act"]
			batch.ahis_next = self.buf[self.buf.next(indices)].info["historical_act"]
		
		batch.obs_delayed_step_num = batch.info["obs_delayed_step_num"]
		for k in list(batch.keys()): 
			if k not in keeped_keys:
				batch.pop(k)
		return batch

	def _pre_update_process(self, batch, burnin_batch=None):
		#保留a-in，c-in等训练真正使用的数据
		""" Pre-update process
		including merging history, obs_pred, obs_encode ... and removing some keys 
		only keep keys used in update
		input keys: ["dobs", "dobs_next", "oobs", "oobs_next", "ahis_cur", "ahis_next", "act", "rew", "done"]
		output keys:
			"a_in_cur", "a_in_next", 
			"c_in_online_cur", "c_in_online_next", "c_in_cur",
			"done", "rew", "act", "valid_mask", "terminated"
			(only when obs_pred) "pred_out_cur", "oobs", 
			(only when obs_encode)
			(only when sac) "logprob_online_cur", "logprob_online_next"
			ps. the key with "online" is with gradient
		"""
		keeped_keys = ["a_in_cur", "a_in_next", "c_in_cur", "c_in_online_cur", "c_in_online_next", "logprob_online_cur", "logprob_online_next", "done", "rew", "act", "valid_mask", "terminated"]
		batch.to_torch(device=self.cfg.device, dtype=torch.float32)
		if self._burnin_num(): burnin_batch.to_torch(device=self.cfg.device, dtype=torch.float32)
		pre_sz = list(batch["done"].shape)
		

		# actor - 1. obs base
		if self.global_cfg.actor_input.obs_type == "normal": 
			batch.a_in_cur = batch.dobs
			batch.a_in_next = batch.dobs_next
			if self._burnin_num(): burnin_batch.a_in = burnin_batch.dobs # only need a_in since we dont dont forward twice, the last state of cur would be used in next
		elif self.global_cfg.actor_input.obs_type == "oracle": 
			batch.a_in_cur  = batch.oobs
			batch.a_in_next = batch.oobs_next
			if self._burnin_num(): burnin_batch.a_in  = burnin_batch.oobs
		else:
			raise ValueError("unknown obs_type: {}".format(self.global_cfg.actor_input.obs_type))

		# actor - 2. others
		if self.global_cfg.actor_input.history_merge_method == "none":
			# TODO seems that the obs_pred and obs_encode can be outside
			if self.global_cfg.actor_input.obs_pred.turn_on:
				keeped_keys += ["pred_out_cur", "oobs"]
				batch.pred_in_cur = batch.a_in_cur
				batch.pred_in_next = batch.a_in_next
				batch.pred_out_cur, pred_info_cur = self.pred_net(batch.pred_in_cur)
				batch.pred_out_next, pred_info_next = self.pred_net(batch.pred_in_next)
				
				if self.global_cfg.actor_input.obs_pred.input_type == "obs": # a input
					batch.a_in_cur = batch.pred_out_cur
					batch.a_in_next = batch.pred_out_next
				elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
					batch.a_in_cur = pred_info_cur["feats"]
					batch.a_in_next = pred_info_next["feats"]
				else:
					raise NotImplementedError
				
				if self.global_cfg.actor_input.obs_pred.middle_detach: # detach
					batch.a_in_cur = batch.a_in_cur.detach()
					batch.a_in_next = batch.a_in_next.detach()
				
				if self.global_cfg.actor_input.obs_pred.net_type == "vae": # vae
					keeped_keys += ["pred_info_cur_mu", "pred_info_cur_logvar"]
					batch.pred_info_cur_mu = pred_info_cur["mu"]
					batch.pred_info_cur_logvar = pred_info_cur["logvar"]
				
			if self.global_cfg.actor_input.obs_encode.turn_on:
				keeped_keys += ["encode_oracle_info_cur_mu","encode_oracle_info_cur_logvar", "encode_normal_info_cur_mu", "encode_normal_info_cur_logvar"]
				
				batch.encode_obs_in_cur = batch.a_in_cur
				batch.encode_obs_in_next = batch.a_in_next

				batch.encode_obs_out_cur, encode_obs_info_cur = self.encode_net.normal_encode(batch.encode_obs_in_cur)
				batch.encode_obs_out_next, _ = self.encode_net.normal_encode(batch.encode_obs_in_next)
				batch.encode_oracle_obs_output_cur, encode_oracle_obs_info_cur = self.encode_net.oracle_encode(batch.oobs)
				batch.encode_oracle_obs_output_next, _ = self.encode_net.oracle_encode(batch.oobs_next)
				
				batch.encode_normal_info_cur_mu = encode_obs_info_cur["mu"]
				batch.encode_normal_info_cur_logvar = encode_obs_info_cur["logvar"]
				batch.encode_oracle_info_cur_mu = encode_oracle_obs_info_cur["mu"]
				batch.encode_oracle_info_cur_logvar = encode_oracle_obs_info_cur["logvar"]
				
				if self.global_cfg.actor_input.obs_encode.train_eval_async == True:
					batch.a_in_cur = batch.encode_oracle_obs_output_cur
					batch.a_in_next = batch.encode_oracle_obs_output_next
				elif self.global_cfg.actor_input.obs_encode.train_eval_async == False:
					batch.a_in_cur = batch.encode_obs_out_cur
					batch.a_in_next = batch.encode_obs_out_next
				else:
					raise ValueError("batch error")
				
				if self.global_cfg.actor_input.obs_encode.pred_loss_weight:
					keeped_keys += ["oobs", "pred_obs_output_cur"]
					batch.pred_obs_output_cur, _ = self.encode_net.decode(batch.encode_obs_out_cur)
				
				if self.global_cfg.actor_input.obs_encode.before_policy_detach:
					batch.a_in_cur = batch.a_in_cur.detach()
					batch.a_in_next = batch.a_in_next.detach()	
		#结合历史数据，cat_mlp
		elif self.global_cfg.actor_input.history_merge_method == "cat_mlp":
			if self.global_cfg.history_num > 0: 
				batch.a_in_cur = torch.cat([batch.a_in_cur, batch.ahis_cur.flatten(start_dim=-2)], dim=-1)
				batch.a_in_next = torch.cat([batch.a_in_next, batch.ahis_next.flatten(start_dim=-2)], dim=-1)
			
			if self.global_cfg.actor_input.obs_pred.turn_on:
				keeped_keys += ["pred_out_cur", "oobs"]
				batch.pred_in_cur = batch.a_in_cur
				batch.pred_in_next = batch.a_in_next
				batch.pred_out_cur, pred_info_cur = self.pred_net(batch.pred_in_cur)
				batch.pred_out_next, pred_info_next = self.pred_net(batch.pred_in_next)
				
				if self.global_cfg.actor_input.obs_pred.input_type == "obs": # a input
					batch.a_in_cur = batch.pred_out_cur
					batch.a_in_next = batch.pred_out_next
				elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
					batch.a_in_cur = pred_info_cur["feats"]
					batch.a_in_next = pred_info_next["feats"]
				else:
					raise NotImplementedError
				
				if self.global_cfg.actor_input.obs_pred.middle_detach: # detach
					batch.a_in_cur = batch.a_in_cur.detach()
					batch.a_in_next = batch.a_in_next.detach()
				
				if self.global_cfg.actor_input.obs_pred.net_type == "vae": # vae
					keeped_keys += ["pred_info_cur_mu", "pred_info_cur_logvar"]
					batch.pred_info_cur_mu = pred_info_cur["mu"]
					batch.pred_info_cur_logvar = pred_info_cur["logvar"]
				
			if self.global_cfg.actor_input.obs_encode.turn_on:
				keeped_keys += ["encode_oracle_info_cur_mu","encode_oracle_info_cur_logvar", "encode_normal_info_cur_mu", "encode_normal_info_cur_logvar"]
				#预测加上his，和oracle真实观测
				batch.encode_obs_in_cur = batch.a_in_cur
				batch.encode_obs_in_next = batch.a_in_next

				batch.encode_obs_out_cur, encode_obs_info_cur = self.encode_net.normal_encode(batch.encode_obs_in_cur)
				batch.encode_obs_out_next, _ = self.encode_net.normal_encode(batch.encode_obs_in_next)
				batch.encode_oracle_obs_output_cur, encode_oracle_obs_info_cur = self.encode_net.oracle_encode(batch.oobs)
				batch.encode_oracle_obs_output_next, _ = self.encode_net.oracle_encode(batch.oobs_next)
				
				batch.encode_normal_info_cur_mu = encode_obs_info_cur["mu"]
				batch.encode_normal_info_cur_logvar = encode_obs_info_cur["logvar"]
				batch.encode_oracle_info_cur_mu = encode_oracle_obs_info_cur["mu"]
				batch.encode_oracle_info_cur_logvar = encode_oracle_obs_info_cur["logvar"]
				
				if self.global_cfg.actor_input.obs_encode.train_eval_async == True:
					batch.a_in_cur = batch.encode_oracle_obs_output_cur
					batch.a_in_next = batch.encode_oracle_obs_output_next
				elif self.global_cfg.actor_input.obs_encode.train_eval_async == False:
					batch.a_in_cur = batch.encode_obs_out_cur
					batch.a_in_next = batch.encode_obs_out_next
				else:
					raise ValueError("batch error")
				
				if self.global_cfg.actor_input.obs_encode.pred_loss_weight:
					keeped_keys += ["oobs", "pred_obs_output_cur"]
					batch.pred_obs_output_cur, _ = self.encode_net.decode(batch.encode_obs_out_cur)
				
				if self.global_cfg.actor_input.obs_encode.before_policy_detach:
					batch.a_in_cur = batch.a_in_cur.detach()
					batch.a_in_next = batch.a_in_next.detach()	
		elif self.global_cfg.actor_input.history_merge_method == "stack_rnn":
			if self.global_cfg.history_num > 0: 
				batch.a_in_cur = torch.cat([batch.a_in_cur, batch.ahis_cur[...,-1,:]], dim=-1) # [*, obs_dim+act_dim]
				batch.a_in_next = torch.cat([batch.a_in_next, batch.ahis_next[...,-1,:]], dim=-1)
				if self._burnin_num(): 
					burnin_batch.a_in = torch.cat([burnin_batch.a_in, burnin_batch.ahis_cur[...,-1,:]], dim=-1) # [B, T, obs_dim+act_dim]
			
			if self._burnin_num():
				keeped_keys += ["burnin_a_in", "burnin_remaster_mask"] # mask reused by critic
				batch.burnin_a_in, batch.burnin_remaster_mask = burnin_batch.a_in, burnin_batch.valid_mask

			if self.global_cfg.actor_input.obs_pred.turn_on:
				raise NotImplementedError
				keeped_keys += ["pred_out_cur", "oobs"]
				batch.pred_in_cur = batch.a_in_cur
				batch.pred_in_next = batch.a_in_next
				batch.pred_out_cur, pred_info_cur = self.pred_net(batch.pred_in_cur)
				batch.pred_out_next, pred_info_next = self.pred_net(batch.pred_in_next)
				
				if self.global_cfg.actor_input.obs_pred.input_type == "obs": # a input
					batch.a_in_cur = batch.pred_out_cur
					batch.a_in_next = batch.pred_out_next
				elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
					batch.a_in_cur = pred_info_cur["feats"]
					batch.a_in_next = pred_info_next["feats"]
				else:
					raise NotImplementedError
				
				if self.global_cfg.actor_input.obs_pred.middle_detach: # detach
					batch.a_in_cur = batch.a_in_cur.detach()
					batch.a_in_next = batch.a_in_next.detach()
				
				if self.global_cfg.actor_input.obs_pred.net_type == "vae": # vae
					keeped_keys += ["pred_info_cur_mu", "pred_info_cur_logvar"]
					batch.pred_info_cur_mu = pred_info_cur["mu"]
					batch.pred_info_cur_logvar = pred_info_cur["logvar"]
				
			if self.global_cfg.actor_input.obs_encode.turn_on:
				raise NotImplementedError
				keeped_keys += ["encode_oracle_info_cur_mu","encode_oracle_info_cur_logvar", "encode_normal_info_cur_mu", "encode_normal_info_cur_logvar"]
				
				batch.encode_obs_in_cur = batch.a_in_cur
				batch.encode_obs_in_next = batch.a_in_next

				batch.encode_obs_out_cur, encode_obs_info_cur = self.encode_net.normal_encode(batch.encode_obs_in_cur)
				batch.encode_obs_out_next, _ = self.encode_net.normal_encode(batch.encode_obs_in_next)
				batch.encode_oracle_obs_output_cur, encode_oracle_obs_info_cur = self.encode_net.oracle_encode(batch.oobs)
				batch.encode_oracle_obs_output_next, _ = self.encode_net.oracle_encode(batch.oobs_next)
				
				batch.encode_normal_info_cur_mu = encode_obs_info_cur["mu"]
				batch.encode_normal_info_cur_logvar = encode_obs_info_cur["logvar"]
				batch.encode_oracle_info_cur_mu = encode_oracle_obs_info_cur["mu"]
				batch.encode_oracle_info_cur_logvar = encode_oracle_obs_info_cur["logvar"]
				
				if self.global_cfg.actor_input.obs_encode.train_eval_async == True:
					batch.a_in_cur = batch.encode_oracle_obs_output_cur
					batch.a_in_next = batch.encode_oracle_obs_output_next
				elif self.global_cfg.actor_input.obs_encode.train_eval_async == False:
					batch.a_in_cur = batch.encode_obs_out_cur
					batch.a_in_next = batch.encode_obs_out_next
				else:
					raise ValueError("batch error")
				
				if self.global_cfg.actor_input.obs_encode.pred_loss_weight:
					keeped_keys += ["oobs", "pred_obs_output_cur"]
					batch.pred_obs_output_cur, _ = self.encode_net.decode(batch.encode_obs_out_cur)
				
				if self.global_cfg.actor_input.obs_encode.before_policy_detach:
					batch.a_in_cur = batch.a_in_cur.detach()
					batch.a_in_next = batch.a_in_next.detach()
		elif self.global_cfg.actor_input.history_merge_method == "transformer":
			if self.global_cfg.history_num > 0: 
				batch.a_in_cur = torch.cat([batch.ohis_cur, batch.ahis_cur], dim=-1) # [*, obs_dim+act_dim]
				batch.a_in_next = torch.cat([batch.ohis_next, batch.ahis_next], dim=-1)

			if self.global_cfg.actor_input.obs_pred.turn_on:
				raise NotImplementedError
				keeped_keys += ["pred_out_cur", "oobs"]
				batch.pred_in_cur = batch.a_in_cur
				batch.pred_in_next = batch.a_in_next
				batch.pred_out_cur, pred_info_cur = self.pred_net(batch.pred_in_cur)
				batch.pred_out_next, pred_info_next = self.pred_net(batch.pred_in_next)
				
				if self.global_cfg.actor_input.obs_pred.input_type == "obs": # a input
					batch.a_in_cur = batch.pred_out_cur
					batch.a_in_next = batch.pred_out_next
				elif self.global_cfg.actor_input.obs_pred.input_type == "feat":
					batch.a_in_cur = pred_info_cur["feats"]
					batch.a_in_next = pred_info_next["feats"]
				else:
					raise NotImplementedError
				
				if self.global_cfg.actor_input.obs_pred.middle_detach: # detach
					batch.a_in_cur = batch.a_in_cur.detach()
					batch.a_in_next = batch.a_in_next.detach()
				
				if self.global_cfg.actor_input.obs_pred.net_type == "vae": # vae
					keeped_keys += ["pred_info_cur_mu", "pred_info_cur_logvar"]
					batch.pred_info_cur_mu = pred_info_cur["mu"]
					batch.pred_info_cur_logvar = pred_info_cur["logvar"]
				
			if self.global_cfg.actor_input.obs_encode.turn_on:
				raise NotImplementedError
				keeped_keys += ["encode_oracle_info_cur_mu","encode_oracle_info_cur_logvar", "encode_normal_info_cur_mu", "encode_normal_info_cur_logvar"]
				
				batch.encode_obs_in_cur = batch.a_in_cur
				batch.encode_obs_in_next = batch.a_in_next

				batch.encode_obs_out_cur, encode_obs_info_cur = self.encode_net.normal_encode(batch.encode_obs_in_cur)
				batch.encode_obs_out_next, _ = self.encode_net.normal_encode(batch.encode_obs_in_next)
				batch.encode_oracle_obs_output_cur, encode_oracle_obs_info_cur = self.encode_net.oracle_encode(batch.oobs)
				batch.encode_oracle_obs_output_next, _ = self.encode_net.oracle_encode(batch.oobs_next)
				
				batch.encode_normal_info_cur_mu = encode_obs_info_cur["mu"]
				batch.encode_normal_info_cur_logvar = encode_obs_info_cur["logvar"]
				batch.encode_oracle_info_cur_mu = encode_oracle_obs_info_cur["mu"]
				batch.encode_oracle_info_cur_logvar = encode_oracle_obs_info_cur["logvar"]
				
				if self.global_cfg.actor_input.obs_encode.train_eval_async == True:
					batch.a_in_cur = batch.encode_oracle_obs_output_cur
					batch.a_in_next = batch.encode_oracle_obs_output_next
				elif self.global_cfg.actor_input.obs_encode.train_eval_async == False:
					batch.a_in_cur = batch.encode_obs_out_cur
					batch.a_in_next = batch.encode_obs_out_next
				else:
					raise ValueError("batch error")
				
				if self.global_cfg.actor_input.obs_encode.pred_loss_weight:
					keeped_keys += ["oobs", "pred_obs_output_cur"]
					batch.pred_obs_output_cur, _ = self.encode_net.decode(batch.encode_obs_out_cur)
				
				if self.global_cfg.actor_input.obs_encode.before_policy_detach:
					batch.a_in_cur = batch.a_in_cur.detach()
					batch.a_in_next = batch.a_in_next.detach()
		
		else: 
			raise ValueError("history_merge_method error")

		# get online act from actor，从actor采样动作，用来训练
		act_online, act_online_next, act_online_info = self.get_act_online(batch)
		if "logprob_online_cur" in act_online_info: batch.logprob_online_cur = act_online_info["logprob_online_cur"]
		if "logprob_online_next" in act_online_info: batch.logprob_online_next = act_online_info["logprob_online_next"]

		# critic - 1. obs base，crictic的输入应该是oracle
		if self.global_cfg.critic_input.obs_type == "normal": 
			batch.c_in_cur = batch.dobs
			batch.c_in_next = batch.dobs_next
			if self._burnin_num(): burnin_batch.c_in  = burnin_batch.dobs
		elif self.global_cfg.critic_input.obs_type == "oracle": 
			batch.c_in_cur  = batch.oobs
			batch.c_in_next = batch.oobs_next
			if self._burnin_num(): burnin_batch.c_in  = burnin_batch.oobs
		else:
			raise ValueError("unknown obs_type: {}".format(self.global_cfg.critic_input.obs_type))

		# critic - 2. merge act
		batch.c_in_online_cur = torch.cat([batch.c_in_cur, act_online], dim=-1)
		batch.c_in_online_next = torch.cat([batch.c_in_next, act_online_next], dim=-1)
		batch.c_in_cur = torch.cat([batch.c_in_cur, batch.act], dim=-1)
		if self._burnin_num(): burnin_batch.c_in = torch.cat([burnin_batch.c_in, burnin_batch.act], dim=-1) # [B, T, obs_dim+act_dim]

		# critic - 3. merge act history，critic是否需要结合原来的历史动作数据
		if self.cfg.global_cfg.critic_input.history_merge_method == "none":
			pass
		elif self.cfg.global_cfg.critic_input.history_merge_method == "cat_mlp":
			if self.global_cfg.history_num > 0:
				batch.c_in_online_cur = torch.cat([batch.c_in_online_cur, batch.ahis_cur.flatten(start_dim=-2)], dim=-1)
				batch.c_in_online_next = torch.cat([batch.c_in_online_next, batch.ahis_next.flatten(start_dim=-2)], dim=-1)
				batch.c_in_cur = torch.cat([batch.c_in_cur, batch.ahis_cur.flatten(start_dim=-2)], dim=-1)
		elif self.cfg.global_cfg.critic_input.history_merge_method == "stack_rnn":
			# TODO the state for critic varies a lot across critic1, critic2, criticold, so we cannot save the state then process
			assert self.global_cfg.history_num == 1, "in stack_rnn, history_num must be 1"
			assert 1, "RNN layer must be 1 since in current implementation we can only get state of last layer"
			keeped_keys += ["preinput"]
			batch.c_in_online_cur = torch.cat([batch.c_in_online_cur, batch.ahis_cur[...,-1,:]], dim=-1)
			batch.c_in_online_next = torch.cat([batch.c_in_online_next, batch.ahis_next[...,-1,:]], dim=-1)
			batch.c_in_cur = torch.cat([batch.c_in_cur, batch.ahis_cur[...,-1,:]], dim=-1)
			batch.preinput = batch.c_in_cur
			if self._burnin_num(): # TODO
				burnin_batch.c_in = torch.cat([burnin_batch.c_in, burnin_batch.ahis_cur[...,-1,:]], dim=-1) # [B, T, obs_dim+act_dim+act_dim*act_dim]
			if self._burnin_num():
				keeped_keys += ["burnin_c_in", "burnin_remaster_mask"] # mask reused by critic
				batch.burnin_c_in, batch.burnin_remaster_mask = burnin_batch.c_in, burnin_batch.valid_mask
		else:
			raise ValueError("unknown history_merge_method: {}".format(self.cfg.global_cfg.critic_input.history_merge_method))
		
		# log
		self.record("learn/obs_delayed_step_num", batch.obs_delayed_step_num.mean().item())
		self.record("learn/obs_delayed_step_num_sample", batch.obs_delayed_step_num.flatten()[0].item())

		# only keep res keys
		for k in list(batch.keys()): 
			if k not in keeped_keys: batch.pop(k)
		
		return batch

	def on_evaluate_end(self):
		"""called after a step of evaluation"""
		pass

	def _get_historical_act(self, indices, step, buffer, type=None, device=None):
		""" get historical act
		input [t_0, t_1, ...]
		output [
			[t_0-step, t_0-step+1, ... t_0-1],
			[t_1-step, t_1-step+1, ... t_1-1],
			...
		]
		ps. note that cur step is not included
		ps. the neg step is set to 0.
		:param indices: indices of the batch (B,)
		:param step: the step of the batch. int
		:param buffer: the buffer. 
		:return: historical act (B, step)
		"""
		raise ValueError("Deprecated")
		assert type in ["cat", "stack"], "type must be cat or stack"
		# [t_0-step, t_0-step+1, ... t_0-1, t_0]
		idx_stack_plus1 = utils.idx_stack(indices, buffer, step+1, direction="prev")
		# [t_0-step,   t_0-step+1, ..., t_0-1]
		idx_stack_next = idx_stack_plus1[:, :-1] # (B, step)
		# [t_0-step+1, t_0-step+2, ...,   t_0]
		idx_stack = idx_stack_plus1[:, 1:] # (B, step)
		invalid = (idx_stack_next == idx_stack) # (B, step)
		historical_act = buffer[idx_stack].act # (B, step, act_dim)
		historical_act[invalid] = 0.
		if type == "cat":
			historical_act = historical_act.reshape(historical_act.shape[0], -1) # (B, step*act_dim)
		historical_act = torch.tensor(historical_act, device=device)
		return historical_act

	def adjust_idx_stack(self, idx_stack, adjust_dis, buffer):
		#繁殖buffer越界的方案
		"""
		:param idx_stack: (B, T)
		:param adjust_dis: int
		:param buffer: the buffer
		if the idx_stack start is < adjust_dis to the start of the buffer, then adjust it to the start
		"""
		idx_start = idx_stack[:, 0]
		for _ in range(adjust_dis):
			idx_start = buffer.prev(idx_start)
		idx_to_adjust = idx_start == buffer.prev(idx_start)
		idx_start_to_stack_list = []
		idx_start_to_stack = idx_start.copy()
		for i in range(idx_stack.shape[1]):
			idx_start_to_stack_list.append(idx_start_to_stack)
			idx_start_to_stack = buffer.next(idx_start_to_stack)
		idx_stack_all_adjusted = np.stack(idx_start_to_stack_list, axis=1)
		idx_stack[idx_to_adjust] = idx_stack_all_adjusted[idx_to_adjust]
		return idx_stack

	def get_historical_act(self, indices, step, buffer, type=None, device=None):
		#从buffer中打包历史动作，用来得到history-acts
		# Get the action dimension from the buffer
		act_dim = buffer[0].act.shape[0]

		# Determine the output shape based on the type
		if type == "stack":
			output_shape = (*indices.shape, step, act_dim)
		elif type == "cat":
			output_shape = (*indices.shape, step * act_dim)
		else:
			raise ValueError("Invalid type: choose 'stack' or 'cat'")

		# Create an empty tensor with the output shape
		res = np.zeros(output_shape)

		# Iterate through the input indices and retrieve previous actions
		for i in range(step - 1, -1, -1):  # Reverse loop using a single line
			prev_indices = buffer.prev(indices)
			idx_start = prev_indices == indices
			res_batch_act = buffer[prev_indices].act

			# Handle the case when the requested action is at the start of the buffer
			res_batch_act[idx_start] = 0.

			# Fill the output tensor with the retrieved actions
			if type == "stack":
				res[..., i, :] = res_batch_act
			elif type == "cat":
				res[..., i * act_dim:(i + 1) * act_dim] = res_batch_act

			indices = prev_indices
		
		# Convert the output tensor to a torch tensor
		res = torch.tensor(res, device=device)

		return res

	def _soft_update(self, tgt: nn.Module, src: nn.Module, tau: float) -> None:
		"""Softly update the parameters of target module towards the parameters \
		of source module."""
		#软更新，让目标网络慢慢向源网络靠近
		for tgt_param, src_param in zip(tgt.parameters(), src.parameters()):
			tgt_param.data.copy_(tau * src_param.data + (1 - tau) * tgt_param.data)
				# update the target network
	#burn-in是rnn训练，用来预热rnn的隐藏状态
	def _burnin_num(self):
		if "burnin_num" not in self.cfg.global_cfg: return 0
		if not self.cfg.global_cfg.burnin_num: return 0
		if self.cfg.global_cfg.actor_input.history_merge_method != "stack_rnn": return 0
		burnin_num = self.cfg.global_cfg.burnin_num
		if type(self.cfg.global_cfg.burnin_num) == float:
			burnin_num = int(self.cfg.global_cfg.burnin_num * self.cfg.trainer.batch_seq_len)
		elif type(self.cfg.global_cfg.burnin_num) == int:
			burnin_num = self.cfg.global_cfg.burnin_num
		return burnin_num

	def get_act_online(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Returns the current and next actions to be taken by the agent.

		ps. include all actor forwards during learning

		Motivation: the forward of the actor is across algorithms,
			more importantly, the output results should be used to make 
			critic input

		Returns:
			A tuple of two torch.Tensor objects representing the current and next actions.
		"""
		raise NotImplementedError

	def add_obs_pred_loss(self, batch, combined_loss):
		#添加观测预测的损失
		pred_cfg = self.global_cfg.actor_input.obs_pred
		pred_loss = (batch.pred_out_cur - batch.oobs) ** 2
		pred_loss = apply_mask(pred_loss, batch.valid_mask).mean()
		pred_loss_normed = pred_loss / batch.valid_mask.float().mean()
		combined_loss += pred_loss.mean() * pred_cfg.pred_loss_weight
		self.record("learn/obs_pred/pred_loss", pred_loss.item())
		self.record("learn/obs_pred/pred_loss_normed", pred_loss_normed.item())
		self.record("learn/obs_pred/pred_abs_error", pred_loss.item() ** 0.5)
		self.record("learn/obs_pred/pred_abs_error_normed", pred_loss_normed.item() ** 0.5)
		if pred_cfg.net_type == "vae":
			kl_loss = kl_divergence(
				batch.pred_info_cur_mu,
				batch.pred_info_cur_logvar,
				torch.zeros_like(batch.pred_info_cur_mu),
				torch.zeros_like(batch.pred_info_cur_logvar),
			)
			kl_loss = apply_mask(kl_loss, batch.valid_mask).mean()
			kl_loss_normed = kl_loss / batch.valid_mask.float().mean()
			combined_loss += kl_loss * torch.exp(self.kl_weight_log).detach()
			self.record("learn/obs_pred/loss_kl", kl_loss.item())
			self.record("learn/obs_pred/loss_kl_normed", kl_loss_normed.item())
			if pred_cfg.auto_kl_target:
				kl_weight_loss = - (kl_loss_normed.detach() - pred_cfg.auto_kl_target) * torch.exp(self.kl_weight_log)
				self._auto_kl_optim.zero_grad()
				kl_weight_loss.backward()
				self._auto_kl_optim.step()
				self.record("learn/obs_pred/kl_weight_log", self.kl_weight_log.detach().cpu().item())
				self.record("learn/obs_pred/kl_weight", torch.exp(self.kl_weight_log).detach().cpu().item())
		return combined_loss
	
	def add_obs_encode_loss(self, batch, combined_loss):
		encode_cfg = self.global_cfg.actor_input.obs_encode
		kl_loss = kl_divergence(batch.encode_oracle_info_cur_mu, batch.encode_oracle_info_cur_logvar, batch.encode_normal_info_cur_mu, batch.encode_normal_info_cur_logvar)
		kl_loss = apply_mask(kl_loss, batch.valid_mask).mean()
		kl_loss_normed = kl_loss / batch.valid_mask.float().mean()
		combined_loss += kl_loss * torch.exp(self.kl_weight_log).detach().mean()
		self.record("learn/obs_encode/loss_kl", kl_loss.item())
		self.record("learn/obs_encode/loss_kl_normed", kl_loss_normed.item())

		if encode_cfg.pred_loss_weight:
			pred_loss = (batch.pred_obs_output_cur - batch.oobs) ** 2
			pred_loss = apply_mask(pred_loss, batch.valid_mask).mean()
			self.record("learn/obs_encode/loss_pred", pred_loss.item())
			self.record("learn/obs_encode/abs_error_pred", pred_loss.item() ** 0.5)
			combined_loss += pred_loss * encode_cfg.pred_loss_weight
		
		if encode_cfg.policy_robust_weight:
			dist = Normal(
				batch.encode_normal_info_cur_mu, 
				torch.exp(0.5*batch.encode_normal_info_cur_logvar)
			)
			z_1, z_2 = dist.sample(), dist.sample()
			(a_mu_1, a_var_1), _ = self.actor(z_1, None)
			(a_mu_2, a_var_2), _ = self.actor(z_2, None)
			robust_loss = (a_mu_1 - a_mu_2) ** 2 + (a_var_1.sqrt() - a_var_2.sqrt()) ** 2
			robust_loss = apply_mask(robust_loss, batch.valid_mask).mean()
			robust_loss_normed = robust_loss / batch.valid_mask.float().mean()
			self.record("learn/obs_encode/loss_robust", robust_loss.item())
			self.record("learn/obs_encode/loss_robust_normed", robust_loss_normed.item())
			combined_loss += robust_loss * encode_cfg.policy_robust_weight

		if encode_cfg.auto_kl_target:
			if self.global_cfg.debug.auto_kl_use_log:  # in paper
				kl_weight_loss = - self.kl_weight_log * (
					torch.log10(torch.clamp(kl_loss_normed.detach(), 1e-9, np.inf)) - \
					np.log10(encode_cfg.auto_kl_target)
				)
			else: # previous
				kl_weight_loss = - torch.exp(self.kl_weight_log) * (
					kl_loss_normed.detach() - \
					encode_cfg.auto_kl_target
				) 
			
			if self.global_cfg.debug.auto_kl_divide_act_dim: # in paper
				kl_weight_loss = kl_weight_loss / self.actor.act_num
			
			self._auto_kl_optim.zero_grad()
			kl_weight_loss.backward()
			self._auto_kl_optim.step()
			self.record("learn/obs_encode/kl_weight_log", self.kl_weight_log.detach().cpu().item())
			self.record("learn/obs_encode/kl_weight", torch.exp(self.kl_weight_log).detach().cpu().item())
		return combined_loss
		
# algorithms
# TODO:有关于q_uncertainty_weight的计算
class Q_SACRunner(TD3SACRunner):
	ALGORITHM = "q_sac"
	
	def _build_dbde_cond(self, batch):
		"""Construct DBDE condition as delayed obs + history actions before any pred/encode override."""
		cond = batch.dobs
		if self.global_cfg.history_num > 0:
			if self.cfg.global_cfg.actor_input.history_merge_method == "stack_rnn":
				his_act = batch.ahis_cur[...,-1,:]
			else: # cat_mlp/none
				his_act = batch.ahis_cur.flatten(start_dim=-2)
			cond = torch.cat([cond, his_act], dim=-1)
		return cond

	def update_dbde(self,batch):
		self.dbde_optimizer.zero_grad()
		dbde_cond = self._build_dbde_cond(batch)
		dbde_loss_dict = self.dbde.training_loss(batch.oobs, dbde_cond) 
		dbde_loss = dbde_loss_dict["loss"]
		dbde_loss.backward()
		self.dbde_optimizer.step()
	def update_critic(self, batch):
		# cal target_q
		pre_sz = list(batch.done.shape)
		
		if self.cfg.global_cfg.debug.use_terminated_mask_for_value:
			value_mask = batch.terminated
		else:
			value_mask = batch.done
		
		# target
		with torch.no_grad():
			if self.cfg.global_cfg.critic_input.history_merge_method == "stack_rnn":
				v_next_1 = forward_with_preinput(batch.c_in_online_next, batch.preinput, self.critic1_old, "next")
				v_next_2 = forward_with_preinput(batch.c_in_online_next, batch.preinput, self.critic2_old, "next")
			else:
				v_next_1 = self.critic1_old(batch.c_in_online_next, None)[0]
				v_next_2 = self.critic2_old(batch.c_in_online_next, None)[0]
			v_next = torch.min(
				v_next_1, v_next_2
			).reshape(*pre_sz) - self._log_alpha.exp().detach() * batch.logprob_online_next.reshape(*pre_sz)
			target_q = batch.rew + self.cfg.policy.gamma * (1 - value_mask.int()) * v_next
		
		# cur
		if self.cfg.global_cfg.critic_input.history_merge_method == "stack_rnn":
			v_cur_1 = forward_with_preinput(batch.c_in_cur, batch.preinput, self.critic1, "cur")
			v_cur_2 = forward_with_preinput(batch.c_in_cur, batch.preinput, self.critic2, "cur")
		else:
			v_cur_1 = self.critic1(batch.c_in_cur, None)[0]
			v_cur_2 = self.critic2(batch.c_in_cur, None)[0]
		critic_loss = F.mse_loss(v_cur_1.reshape(*pre_sz), target_q, reduce=False) + \
			F.mse_loss(v_cur_2.reshape(*pre_sz), target_q, reduce=False)

		# add sirnn extra loss
		if self.cfg.global_cfg.critic_input.history_merge_method == "stack_rnn" \
			and self.cfg.global_cfg.critic_input.bi_or_si_rnn == "both":
			v_cur_sirnn_1 = forward_with_preinput(batch.c_in_cur, batch.preinput, self.critic_sirnn_1, "cur")
			v_cur_sirnn_2 = forward_with_preinput(batch.c_in_cur, batch.preinput, self.critic_sirnn_2, "cur")
			sirnn_loss = F.mse_loss(v_cur_sirnn_1.reshape(*pre_sz), target_q, reduce=False) + \
				F.mse_loss(v_cur_sirnn_2.reshape(*pre_sz), target_q, reduce=False)
			critic_loss += sirnn_loss
			self.record("learn/loss_critic_before_sirnn", critic_loss.detach().mean().item())
			self.record("learn/loss_critic_sirnn", sirnn_loss.detach().mean().item())

		# use mask
		critic_loss = apply_mask(critic_loss, batch.valid_mask).mean()
		self.record("learn/loss_critic", critic_loss.item())
		self.record("learn/loss_critic_normed", critic_loss.item()/batch.valid_mask.float().mean().item())
		self.record("learn/valid_mask_ratio", batch.valid_mask.float().mean())

		self.critic1_optim.zero_grad()
		self.critic2_optim.zero_grad()
		if self.cfg.global_cfg.critic_input.history_merge_method == "stack_rnn" \
			and self.cfg.global_cfg.critic_input.bi_or_si_rnn == "both":
			self.critic_sirnn_1_optim.zero_grad()
			self.critic_sirnn_2_optim.zero_grad()
		critic_loss.backward()
		self.critic1_optim.step()
		self.critic2_optim.step()
		if self.cfg.global_cfg.critic_input.history_merge_method == "stack_rnn" \
			and self.cfg.global_cfg.critic_input.bi_or_si_rnn == "both":
			self.critic_sirnn_1_optim.step()
			self.critic_sirnn_2_optim.step()

		return {
			"critic_loss": critic_loss.cpu().item()
		}

	def update_actor(self, batch):
		res_info = {}
		combined_loss = 0.
		# TODO:batch.q_uncertainty_weight,need to be finished from diffusion model
		### actor loss，
		# TODO：加入q_un的计算
		q_uncertainty_weight = 1.0
		if self.diffusion_model:
			num_dbde_samples = 10
			dbde_cond = self._build_dbde_cond(batch)
			dbde_output = self.dbde.sample(dbde_cond, num_dbde_samples)  # [B*num_dbde_samples, obs_dim]
			if self.cfg.global_cfg.critic_input.history_merge_method != "stack_rnn":
				obs_dim = self.env.observation_space.shape[0]
				# 复制 critic 输入，再用 DBDE 生成的 obs 替换 obs 部分，保持动作/历史不变
				c_in_expanded = batch.c_in_online_cur.repeat_interleave(num_dbde_samples, dim=0).clone()
				c_in_expanded[..., :obs_dim] = dbde_output
				q_dbde, _ = self.critic1(c_in_expanded, None)  # [B*num_dbde_samples, 1]
				q_dbde = q_dbde.view(batch.c_in_online_cur.shape[0], num_dbde_samples, -1)
				q_var = q_dbde.var(dim=1, unbiased=False).mean(dim=1, keepdim=True)  # [B,1]
				q_uncertainty_weight = 1.0 / (1.0 + q_var)  # 方差越大，权重越小
				q_uncertainty_weight = q_uncertainty_weight.detach()
			else:
				# RNN critic 情况下暂时无法对 preinput 做替换，先用常数权重
				q_uncertainty_weight = 1.0
		if self.cfg.global_cfg.critic_input.history_merge_method == "stack_rnn":
			if self.cfg.global_cfg.critic_input.bi_or_si_rnn == "both":
				current_q1a = forward_with_preinput(batch.c_in_online_cur, batch.preinput, self.critic_sirnn_1, "cur")
				current_q2a = forward_with_preinput(batch.c_in_online_cur, batch.preinput, self.critic_sirnn_2, "cur")
			else:
				current_q1a = forward_with_preinput(batch.c_in_online_cur, batch.preinput, self.critic1, "cur")
				current_q2a = forward_with_preinput(batch.c_in_online_cur, batch.preinput, self.critic2, "cur")
		else:#c_in_online_cur是obs—t+a-online-sample
			current_q1a, _ = self.critic1(batch.c_in_online_cur, None)
			current_q2a, _ = self.critic2(batch.c_in_online_cur, None)
		actor_loss = q_uncertainty_weight * (self._log_alpha.exp().detach() * batch.logprob_online_cur - torch.min(current_q1a, current_q2a))
		actor_loss = apply_mask(actor_loss, batch.valid_mask).mean()
		combined_loss += actor_loss
		#把actorloss记录下来
		self.record("learn/loss_actor", actor_loss.item())
		self.record("learn/loss_actor_normed", actor_loss.item()/batch.valid_mask.float().mean().item())

		# add obs_pred loss，预测
		if self.global_cfg.actor_input.obs_pred.turn_on:
			combined_loss = self.add_obs_pred_loss(batch, combined_loss)
		
		# add obs_encode loss，编码
		if self.global_cfg.actor_input.obs_encode.turn_on:
			combined_loss = self.add_obs_encode_loss(batch, combined_loss)

		# backward and optim，用combined loss对actor和预测模块进行方向传播，我们应该是使用actor_input.obs_encode.turn_on:，仅编码不做预测
		self.actor_optim.zero_grad()
		if self.global_cfg.actor_input.obs_pred.turn_on: self._pred_optim.zero_grad()
		if self.global_cfg.actor_input.obs_encode.turn_on: self._encode_optim.zero_grad()
		combined_loss.backward()
		if self.global_cfg.actor_input.obs_pred.turn_on: self._pred_optim.step()
		if self.global_cfg.actor_input.obs_encode.turn_on: self._encode_optim.step()
		self.actor_optim.step()

		# update alpha (use batch.logprob_online_cur)
		# 更新alpha的，不需要进行修改
		if self._is_auto_alpha:
			if self.global_cfg.debug.use_log_alpha_for_mul_logprob:
				alpha_mul = self._log_alpha
			else:
				alpha_mul = self._log_alpha.exp()
			
			if self.global_cfg.debug.entropy_mask_loss_renorm:
				cur_entropy = - apply_mask(batch.logprob_online_cur.detach(), batch.valid_mask).mean() / batch.valid_mask.mean() # (*, 1)
				alpha_loss = - alpha_mul * (self._target_entropy - cur_entropy)
			else:
				cur_entropy = - batch.logprob_online_cur.detach() # (*, 1)
				alpha_loss = - alpha_mul * apply_mask(self._target_entropy-cur_entropy, batch.valid_mask) # (*, 1)
				alpha_loss = alpha_loss.mean()
			
			self._alpha_optim.zero_grad()
			alpha_loss.backward()
			self._alpha_optim.step()
			self.record("learn/alpha_loss_normed", alpha_loss.item() / batch.valid_mask.float().mean().item())
			self.record("learn/alpha", self._log_alpha.exp().detach().cpu().item())
			self.record("learn/log_alpha", self._log_alpha.detach().cpu().item())
			self.record("learn/entropy", cur_entropy.mean().cpu().item())
			self.record("learn/entropy_target", self._target_entropy)
		
		return {
			"actor_loss": actor_loss.cpu().item()
		}

	def get_act_online(self, batch):
		# actor_input.history_merge_method == “cat_mlp”，a_in_cur是merged obs+actions，actor输出是均值和方差（高斯输出
		(mu, var), _ = self.actor(batch.a_in_cur, None)
		act_online_cur, logprob_online_cur = self.actor.sample_act(mu, var)
		#act的选择以及对应的对数概率
		# next
		with torch.no_grad():
			#下一个状态的onlie——action不需要传播梯度
			(mu, var), _ = self.actor(batch.a_in_next, state=None)
			act_online_next, logprob_online_next = self.actor.sample_act(mu, var)
		return act_online_cur, act_online_next, {
			"logprob_online_cur": logprob_online_cur,
			"logprob_online_next": logprob_online_next
		}
	#更新critic的简单mse优化器，不需要修改
	def _mse_optimizer(self,
			batch: Batch, critic: torch.nn.Module, optimizer: torch.optim.Optimizer
		) -> Tuple[torch.Tensor, torch.Tensor]:
		"""A simple wrapper script for updating critic network."""
		weight = getattr(batch, "weight", 1.0)
		current_q, critic_state = critic(batch.critic_input_cur_offline)
		target_q = torch.tensor(batch.returns).to(current_q.device)
		td = current_q.flatten() - target_q.flatten()
		critic_loss = (
			(td.pow(2) * weight) * batch.valid_mask.flatten()
		).mean()
		critic_loss = (td.pow(2) * weight)
		critic_loss = critic_loss.mean()
		optimizer.zero_grad()
		critic_loss.backward()
		optimizer.step()
		return td, critic_loss
	
	def _init_sac_alpha(self):
		"""
		init self._log_alpha, self._alpha_optim, self._is_auto_alpha, self._target_entropy
		"""
		cfg = self.cfg
		if isinstance(cfg.policy.alpha, Iterable):
			self._is_auto_alpha = True
			self._target_entropy, self._log_alpha, self._alpha_optim = cfg.policy.alpha
			if type(self._target_entropy) == str and self._target_entropy == "neg_act_num":
				self._target_entropy = - np.prod(self.env.action_space.shape)
			elif type(self._target_entropy) == float:
				self._target_entropy = torch.tensor(self._target_entropy).to(self.device)
			else: 
				raise ValueError("Invalid target entropy type.")
			assert cfg.policy.alpha[1].shape == torch.Size([1]) and cfg.policy.alpha[1].requires_grad
			self._alpha_optim = self._alpha_optim([self._log_alpha])
		elif isinstance(cfg.policy.alpha, float):
			self._is_auto_alpha = False
			self._log_alpha = cfg.policy.alpha # here, the cfg alpha is actually log_alpha
		else: 
			raise ValueError("Invalid alpha type.")
