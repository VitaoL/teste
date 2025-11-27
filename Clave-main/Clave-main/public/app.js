const fallbackTeachers = [
  {
    id: 1,
    nome: 'Ana Melo',
    instrumento: 'Violão e Canto',
    preco: 90,
    estilos: ['MPB', 'Pop'],
    bio: 'Cantora e violonista há 10 anos, com foco em repertório brasileiro e técnicas de interpretação.',
    foto: 'https://images.unsplash.com/photo-1487412947147-5cebf100ffc2?auto=format&fit=crop&w=900&q=60',
    video: 'https://www.youtube.com/embed/5qap5aO4i9A'
  },
  {
    id: 2,
    nome: 'Lucas Prado',
    instrumento: 'Piano e Teoria',
    preco: 120,
    estilos: ['Clássico', 'Jazz'],
    bio: 'Pianista clássico com 8 anos de docência. Aulas personalizadas para iniciantes e avançados.',
    foto: 'https://images.unsplash.com/photo-1489424731084-a5d8b219a5bb?auto=format&fit=crop&w=900&q=60',
    video: 'https://www.youtube.com/embed/4Tr0otuiQuU'
  },
  {
    id: 3,
    nome: 'Carla Nogueira',
    instrumento: 'Guitarra',
    preco: 85,
    estilos: ['Rock', 'Blues'],
    bio: 'Guitarrista de estúdio. Foco em improvisação, timbres e repertório moderno.',
    foto: 'https://images.unsplash.com/photo-1492562080023-ab3db95bfbce?auto=format&fit=crop&w=900&q=60',
    video: null
  }
];

const state = {
  user: null,
  teachers: [],
  selected: null,
  filter: ''
};

const els = {};

window.addEventListener('DOMContentLoaded', () => {
  cacheElements();
  bindEvents();
  hydrateUser();
  loadTeachers();
});

function cacheElements() {
  els.grid = document.getElementById('teachers-grid');
  els.list = document.getElementById('teachers-list');
  els.metricTeachers = document.getElementById('metric-teachers');
  els.profileDrawer = document.getElementById('profile-drawer');
  els.profileName = document.getElementById('profile-name');
  els.profileInstrument = document.getElementById('profile-instrument');
  els.profileBio = document.getElementById('profile-bio');
  els.profilePhoto = document.getElementById('profile-photo');
  els.profilePrice = document.getElementById('profile-price');
  els.profileStyles = document.getElementById('profile-styles');
  els.profileVideo = document.getElementById('profile-video');
  els.videoWrapper = document.getElementById('video-wrapper');
  els.btnBook = document.getElementById('btn-book');
  els.bookingModal = document.getElementById('booking-modal');
  els.bookingTeacher = document.getElementById('booking-teacher');
  els.bookingPrice = document.getElementById('booking-price');
  els.bookingForm = document.getElementById('booking-form');
  els.dashboard = document.getElementById('dashboard');
}

function bindEvents() {
  document.getElementById('btn-search').addEventListener('click', () => {
    state.filter = document.getElementById('main-search').value.trim();
    renderTeachers();
  });

  document.getElementById('filter-chips').addEventListener('click', (e) => {
    if (e.target.dataset.filter !== undefined) {
      document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
      e.target.classList.add('active');
      state.filter = e.target.dataset.filter;
      renderTeachers();
    }
  });

  document.querySelectorAll('.pill').forEach(pill => {
    pill.addEventListener('click', () => {
      document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
      pill.classList.add('active');
      const view = pill.dataset.view;
      els.grid.style.display = view === 'cards' ? 'grid' : 'none';
      els.list.style.display = view === 'list' ? 'flex' : 'none';
    });
  });

  document.getElementById('close-profile').addEventListener('click', closeDrawer);
  els.profileDrawer.addEventListener('click', (e) => {
    if (e.target === els.profileDrawer) closeDrawer();
  });

  document.getElementById('login-btn').addEventListener('click', () => openModal('login-modal'));
  document.getElementById('register-btn').addEventListener('click', () => openModal('register-modal'));
  document.querySelectorAll('[data-close]').forEach(btn => btn.addEventListener('click', closeModals));
  document.getElementById('logout-btn').addEventListener('click', logout);

  document.getElementById('register-type').addEventListener('change', (e) => {
    document.getElementById('professor-extra').style.display = e.target.value === 'professor' ? 'block' : 'none';
  });

  document.getElementById('btn-dashboard').addEventListener('click', () => {
    if (!state.user) return;
    els.dashboard.style.display = 'block';
    document.getElementById('dashboard').scrollIntoView({ behavior: 'smooth' });
  });

  document.querySelectorAll('.dash-tab').forEach(tab => tab.addEventListener('click', () => switchTab(tab)));
  document.getElementById('form-dados').addEventListener('submit', (e) => {
    e.preventDefault();
    alert('Dados salvos localmente.');
  });

  document.getElementById('login-form').addEventListener('submit', handleLogin);
  document.getElementById('register-form').addEventListener('submit', handleRegister);
  els.bookingForm.addEventListener('submit', handleBooking);

  els.btnBook?.addEventListener('click', () => {
    if (!state.selected) return;
    els.bookingTeacher.textContent = state.selected.nome;
    els.bookingPrice.value = state.selected.preco;
    openModal('booking-modal');
  });

  els.bookingModal?.addEventListener('click', (e) => {
    if (e.target === els.bookingModal) closeModals();
  });
}

async function loadTeachers() {
  try {
    const resp = await fetch('/api/professores');
    if (!resp.ok) throw new Error('falha na API');
    const data = await resp.json();
    state.teachers = data.map(normalizeTeacher);
  } catch (err) {
    console.warn('Usando fallback de professores:', err.message);
    state.teachers = fallbackTeachers.map(normalizeTeacher);
  }
  renderTeachers();
}

function normalizeTeacher(item) {
  return {
    id: item.id,
    nome: item.nome || item.name,
    instrumento: item.instrumento || item.instrument || (item.instruments ? item.instruments.join(', ') : 'Música'),
    preco: Number(item.preco || item.preco_hora || item.valor_base_aula || item.price || 0),
    estilos: item.styles || item.estilos || item.estilos || [],
    bio: item.bio || item.biografia || item.sobre || '',
    foto: item.foto || item.image || item.foto_url || item.photoUrl || `https://via.placeholder.com/640x360?text=${(item.nome || item.name || 'Professor')}`,
    video: item.video || item.videoUrl || item.link_video_demo || null
  };
}

function renderTeachers() {
  const filter = state.filter.toLowerCase();
  const list = state.teachers.filter(t => {
    if (!filter) return true;
    return t.nome.toLowerCase().includes(filter) || t.instrumento.toLowerCase().includes(filter) || (Array.isArray(t.estilos) && t.estilos.join(',').toLowerCase().includes(filter));
  });

  els.metricTeachers.textContent = list.length;
  els.grid.innerHTML = '';
  els.list.innerHTML = '';

  if (list.length === 0) {
    els.grid.innerHTML = '<div class="empty-state">Nenhum professor encontrado para esse filtro.</div>';
    return;
  }

  list.forEach(t => {
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `
      <img src="${t.foto}" alt="${t.nome}">
      <h3>${t.nome}</h3>
      <p class="muted">${t.instrumento}</p>
      <span class="badge">${t.estilos && t.estilos.length ? t.estilos.join(', ') : 'Multiestilo'}</span>
      <p class="price">R$ ${t.preco}/hora</p>
    `;
    card.onclick = () => openDrawer(t);
    els.grid.appendChild(card);

    const row = document.createElement('div');
    row.className = 'list-item';
    row.innerHTML = `
      <img src="${t.foto}" alt="${t.nome}">
      <div>
        <h4>${t.nome}</h4>
        <p class="muted">${t.instrumento}</p>
        <span class="badge">R$ ${t.preco}/h</span>
      </div>
    `;
    row.onclick = () => openDrawer(t);
    els.list.appendChild(row);
  });
}

function openDrawer(teacher) {
  state.selected = teacher;
  els.profileName.textContent = teacher.nome;
  els.profileInstrument.textContent = teacher.instrumento;
  els.profileBio.textContent = teacher.bio || 'Professor ainda não cadastrou uma biografia.';
  els.profilePhoto.src = teacher.foto;
  els.profilePrice.textContent = teacher.preco.toFixed(2);
  els.profileStyles.innerHTML = '';
  (teacher.estilos || []).forEach(est => {
    const pill = document.createElement('span');
    pill.className = 'pill';
    pill.textContent = est;
    els.profileStyles.appendChild(pill);
  });
  if (!teacher.estilos || teacher.estilos.length === 0) {
    els.profileStyles.innerHTML = '<span class="pill">Multiestilo</span>';
  }

  if (teacher.video) {
    els.videoWrapper.style.display = 'block';
    els.profileVideo.src = teacher.video;
  } else {
    els.videoWrapper.style.display = 'none';
    els.profileVideo.src = '';
  }

  els.profileDrawer.style.display = 'flex';
}

function closeDrawer() {
  els.profileDrawer.style.display = 'none';
}

function openModal(id) {
  document.getElementById(id).style.display = 'flex';
}
function closeModals() {
  document.querySelectorAll('.modal').forEach(m => m.style.display = 'none');
}

function hydrateUser() {
  const saved = localStorage.getItem('clave:user');
  if (saved) {
    state.user = JSON.parse(saved);
    applyUser();
  }
}

function applyUser() {
  const auth = document.getElementById('auth-area');
  const user = document.getElementById('user-area');
  if (state.user) {
    auth.style.display = 'none';
    user.style.display = 'flex';
    document.getElementById('user-name').textContent = state.user.name;
    document.getElementById('user-role').textContent = state.user.type === 'professor' ? 'Professor' : 'Aluno';
    document.getElementById('dash-name').textContent = state.user.name;
    document.getElementById('dash-role').textContent = state.user.type === 'professor' ? 'Professor' : 'Aluno';
    document.getElementById('input-dash-name').value = state.user.name;
    document.getElementById('input-dash-email').value = state.user.email;
  } else {
    auth.style.display = 'flex';
    user.style.display = 'none';
    els.dashboard.style.display = 'none';
  }
}

async function handleLogin(e) {
  e.preventDefault();
  const email = document.getElementById('login-email').value;
  const password = document.getElementById('login-password').value;
  try {
    const resp = await fetch('/api/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ email, password }) });
    if (!resp.ok) throw new Error('Falha no login');
    const data = await resp.json();
    state.user = data;
    localStorage.setItem('clave:user', JSON.stringify(data));
    applyUser();
    closeModals();
  } catch (err) {
    alert('Não foi possível entrar. Usando login local para testar.');
    state.user = { id: Date.now(), name: email.split('@')[0], email, type: 'aluno' };
    localStorage.setItem('clave:user', JSON.stringify(state.user));
    applyUser();
    closeModals();
  }
}

async function handleRegister(e) {
  e.preventDefault();
  const type = document.getElementById('register-type').value;
  const payload = {
    name: document.getElementById('register-name').value,
    email: document.getElementById('register-email').value,
    password: document.getElementById('register-password').value,
    type,
    bio: document.getElementById('reg-bio').value,
    price: document.getElementById('reg-price').value,
    instruments: document.getElementById('reg-instruments').value.split(',').map(s => s.trim()).filter(Boolean),
    styles: document.getElementById('reg-styles').value.split(',').map(s => s.trim()).filter(Boolean),
    videoUrl: document.getElementById('reg-video').value,
    photoUrl: document.getElementById('reg-photo').value
  };

  try {
    const resp = await fetch('/api/register', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    if (!resp.ok) throw new Error('Falha no cadastro');
    const data = await resp.json();
    state.user = data;
    localStorage.setItem('clave:user', JSON.stringify(data));
    applyUser();
    closeModals();
    alert('Conta criada com sucesso!');
  } catch (err) {
    console.warn('Cadastro local por fallback:', err.message);
    const localUser = { id: Date.now(), name: payload.name, email: payload.email, type };
    state.user = localUser;
    localStorage.setItem('clave:user', JSON.stringify(localUser));
    applyUser();
    closeModals();
  }
}

function logout() {
  localStorage.removeItem('clave:user');
  state.user = null;
  applyUser();
}

function switchTab(tab) {
  document.querySelectorAll('.dash-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.dash-panel').forEach(p => p.classList.remove('active'));
  tab.classList.add('active');
  document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
}

async function handleBooking(e) {
  e.preventDefault();
  if (!state.user) {
    alert('Faça login para agendar.');
    openModal('login-modal');
    return;
  }
  try {
    const body = {
      teacherId: state.selected?.id,
      studentId: state.user.id,
      date: document.getElementById('booking-date').value,
      time: document.getElementById('booking-time').value,
      price: document.getElementById('booking-price').value
    };
    const resp = await fetch('/api/payment', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!resp.ok) throw new Error('Falha na API');
    alert('Aula agendada com sucesso!');
    closeModals();
    closeDrawer();
  } catch (err) {
    alert('Agendamento salvo localmente para demonstração.');
    closeModals();
    closeDrawer();
  }
}
