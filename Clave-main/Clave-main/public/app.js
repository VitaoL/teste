// public/app.js

let currentUser = null;
let selectedTeacher = null;

document.addEventListener('DOMContentLoaded', () => {
    checkLogin();
    loadMarketplace(); // Carrega PROFESSORES

    // Configurar Filtros
    const categoryTags = document.querySelectorAll('.category-tag');
    for (const tag of categoryTags) {
        tag.addEventListener('click', () => {
            for (const t of categoryTags) t.classList.remove('active');
            tag.classList.add('active');
            loadMarketplace(tag.dataset.filter);
        });
    }

    // Configurar Busca
    document.getElementById('btn-search').addEventListener('click', () => {
        loadMarketplace(document.getElementById('main-search').value);
    });
});

function checkLogin() {
    const user = localStorage.getItem('currentUser');
    if (user) {
        currentUser = JSON.parse(user);
        updateUIForUser();
    }
}

function updateUIForUser() {
    const authButtons = document.getElementById('auth-buttons');
    const userMenu = document.getElementById('user-menu');
    const btnDashboard = document.getElementById('btn-dashboard');

    if (currentUser) {
        authButtons.style.display = 'none';
        userMenu.style.display = 'flex';
        btnDashboard.style.display = 'block';
        document.getElementById('user-name').textContent = currentUser.name.split(' ')[0];
    } else {
        authButtons.style.display = 'flex';
        userMenu.style.display = 'none';
        btnDashboard.style.display = 'none';
    }
}

// --- VITRINE DE PROFESSORES ---
async function loadMarketplace(filter = "") {
    const container = document.getElementById('teachers-list');
    container.innerHTML = '<p>Buscando professores...</p>';

    try {
        const response = await fetch('/api/professores');
        const teachers = await response.json();
        container.innerHTML = '';

        const filtered = teachers.filter(t => {
            if (!filter) return true;
            const search = filter.toLowerCase();
            const inName = t.name.toLowerCase().includes(search);
            const inInst = t.instruments && t.instruments.some(i => i.toLowerCase().includes(search));
            return inName || inInst;
        });

        if (filtered.length === 0) {
            container.innerHTML = '<p>Nenhum professor encontrado.</p>';
            return;
        }

        for (const teacher of filtered) {
            const card = document.createElement('div');
            card.className = 'gig-card';
            card.onclick = () => openTeacherProfile(teacher.id);

            const imgUrl = teacher.image || `https://via.placeholder.com/300x200?text=${teacher.name}`;
            const instrumentText = teacher.instruments && teacher.instruments.length > 0 
                ? teacher.instruments.join(', ') 
                : 'Música';

            card.innerHTML = `
                <div class="gig-image" style="background-image: url('${imgUrl}'); height: 220px;"></div>
                <div class="gig-details">
                    <div class="gig-author">
                        <span class="author-name" style="font-size: 1.1rem;">${teacher.name}</span>
                    </div>
                    <h3 class="gig-title" style="color: #666; font-size: 0.95rem; margin-top: 5px;">
                        Ensina: <strong>${instrumentText}</strong>
                    </h3>
                    <div class="gig-rating">
                        <i class="fas fa-star"></i> ${teacher.rating || '5.0'} <span>(${teacher.reviews || 0})</span>
                    </div>
                </div>
                <div class="gig-footer">
                    <div class="gig-price">AULA A PARTIR DE <br><strong>R$ ${teacher.price}/hora</strong></div>
                </div>
            `;
            container.appendChild(card);
        }
    } catch (error) {
        console.error(error);
        container.innerHTML = '<p>Erro ao carregar a vitrine.</p>';
    }
}

async function openTeacherProfile(id) {
    try {
        const response = await fetch(`/api/professores/${id}`);
        const teacher = await response.json();
        selectedTeacher = teacher;

        document.getElementById('profile-img').src = teacher.image;
        document.getElementById('profile-name').textContent = teacher.name;
        
        const inst = teacher.instruments ? teacher.instruments.join(', ') : '';
        const estilos = teacher.styles ? teacher.styles.join(', ') : '';
        document.getElementById('profile-instruments').innerHTML = `<strong>Ensina:</strong> ${inst} <br> <span style="font-size: 0.9rem; color: #888;">Estilos: ${estilos}</span>`;
        
        document.getElementById('profile-bio').textContent = teacher.bio;
        document.getElementById('profile-price').textContent = `R$ ${teacher.price}`;
        
        const videoFrame = document.getElementById('profile-video');
        if (teacher.link_video_demo) { 
             videoFrame.src = teacher.link_video_demo; 
             videoFrame.parentElement.style.display = 'block';
        } else {
             videoFrame.parentElement.style.display = 'none';
        }

        showPage('teacher-profile');
        
        const btnBook = document.getElementById('btn-book-lesson');
        if (currentUser && currentUser.id === teacher.id && currentUser.type === 'professor') {
            btnBook.style.display = 'none';
        } else {
            btnBook.style.display = 'block';
            btnBook.onclick = () => openPaymentModal();
        }

    } catch (error) {
        alert('Erro ao abrir perfil.');
    }
}

function openPaymentModal() {
    if (!currentUser) {
        alert('Faça login para agendar.');
        document.getElementById('login-modal').style.display = 'flex';
        return;
    }
    document.getElementById('pay-teacher-name').textContent = selectedTeacher.name;
    document.getElementById('pay-amount').textContent = `R$ ${selectedTeacher.price}`;
    document.getElementById('payment-modal').style.display = 'flex';
}

// --- Navegação ---
function showPage(pageId) {
    const sections = ['home-section', 'teacher-profile-section', 'dashboard-section'];
    for (const sec of sections) document.getElementById(sec).style.display = 'none';
    document.getElementById(`${pageId}-section`).style.display = 'block';
    if(pageId === 'home') window.scrollTo(0,0);
}

// Modais
const loginModal = document.getElementById('login-modal');
const registerModal = document.getElementById('register-modal');
const paymentModal = document.getElementById('payment-modal');

document.getElementById('login-btn').onclick = () => loginModal.style.display = 'flex';
document.getElementById('register-btn').onclick = () => registerModal.style.display = 'flex';

for (const btn of document.querySelectorAll('.close-modal')) {
    btn.onclick = function() { this.closest('.modal').style.display = 'none'; }
}
window.onclick = (e) => { if (e.target.classList.contains('modal')) e.target.style.display = 'none'; }

// Logout
document.getElementById('logout-btn').onclick = () => {
    localStorage.removeItem('currentUser');
    currentUser = null;
    updateUIForUser();
    showPage('home');
    location.reload();
};

// Dashboard
document.getElementById('btn-dashboard').onclick = (e) => {
    e.preventDefault();
    if (currentUser) {
        document.getElementById('dash-name').value = currentUser.name;
        document.getElementById('dash-email').value = currentUser.email;
        showPage('dashboard');
    }
};

// --- Submissão de Formulários ---

// Login
document.getElementById('login-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
        const res = await fetch('/api/login', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ 
                email: document.getElementById('login-email').value, 
                password: document.getElementById('login-password').value 
            })
        });
        if (res.ok) {
            const data = await res.json();
            localStorage.setItem('currentUser', JSON.stringify(data));
            currentUser = data;
            updateUIForUser();
            loginModal.style.display = 'none';
        } else { alert('Erro no login'); }
    } catch (err) { console.error(err); }
});

// Cadastro (AGORA ENVIA TODOS OS DADOS)
document.getElementById('register-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const type = document.getElementById('register-type').value;
    const payload = {
        name: document.getElementById('register-name').value,
        email: document.getElementById('register-email').value,
        password: document.getElementById('register-password').value,
        photoUrl: document.getElementById('reg-photo').value, // NOVO
        type: type
    };

    // Se for professor, adiciona os campos extras
    if (type === 'professor') {
        payload.bio = document.getElementById('reg-bio').value;
        payload.price = document.getElementById('reg-price').value;
        payload.videoUrl = document.getElementById('reg-video').value;
        
        // Pega os instrumentos e estilos (assumindo que você manteve os campos no HTML)
        const instInput = document.getElementById('reg-instruments').value;
        payload.instruments = instInput ? instInput.split(',').map(s => s.trim()) : [];
        
        const stylesInput = document.getElementById('reg-styles').value;
        payload.styles = stylesInput ? stylesInput.split(',').map(s => s.trim()) : [];
    }

    try {
        const res = await fetch('/api/register', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        if (res.ok) {
            const data = await res.json();
            localStorage.setItem('currentUser', JSON.stringify(data));
            currentUser = data;
            updateUIForUser();
            registerModal.style.display = 'none';
            alert('Conta criada com sucesso!');
            showPage('dashboard');
        } else { 
            const err = await res.json();
            alert(err.error || 'Erro ao criar conta.'); 
        }
    } catch (err) { console.error(err); }
});

document.getElementById('payment-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
        const res = await fetch('/api/payment', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                teacherId: selectedTeacher.id,
                studentId: currentUser.id,
                date: document.getElementById('pay-date').value,
                time: document.getElementById('pay-time').value,
                price: selectedTeacher.price
            })
        });
        if (res.ok) {
            alert('Agendamento realizado!');
            paymentModal.style.display = 'none';
            showPage('home');
        }
    } catch (err) { console.error(err); }
});

// Função auxiliar (se não tiver no HTML)
window.toggleProfessorFields = function() {
    const type = document.getElementById('register-type').value;
    const fields = document.getElementById('professor-fields');
    if(fields) fields.style.display = (type === 'professor') ? 'block' : 'none';
}