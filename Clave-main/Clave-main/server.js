require('dotenv').config();
const express = require('express');
const path = require('node:path');
const bcrypt = require('bcryptjs');

const ProfessorRepository = require('./src/persistencia/ProfessorRepository');
const AlunoRepository = require('./src/persistencia/AlunoRepository');
const AulaRepository = require('./src/persistencia/AulaRepository');
const Professor = require('./src/dominio/Professor');

const app = express();
const port = 3000;

const professorRepo = new ProfessorRepository();
const alunoRepo = new AlunoRepository();
const aulaRepo = new AulaRepository();

const fallbackTeachers = [
    { id: 1, nome: 'Ana Melo', instrumento: 'Violão e Canto', preco_hora: 90, bio: 'Cantora e violonista com aulas dinâmicas.', foto: 'https://images.unsplash.com/photo-1487412947147-5cebf100ffc2?auto=format&fit=crop&w=900&q=60', link_video_demo: 'https://www.youtube.com/embed/5qap5aO4i9A' },
    { id: 2, nome: 'Lucas Prado', instrumento: 'Piano e Teoria', preco_hora: 120, bio: 'Pianista clássico com foco em técnica e repertório.', foto: 'https://images.unsplash.com/photo-1489424731084-a5d8b219a5bb?auto=format&fit=crop&w=900&q=60', link_video_demo: 'https://www.youtube.com/embed/4Tr0otuiQuU' },
    { id: 3, nome: 'Carla Nogueira', instrumento: 'Guitarra', preco_hora: 85, bio: 'Guitarrista de estúdio, especialista em improvisação.', foto: 'https://images.unsplash.com/photo-1492562080023-ab3db95bfbce?auto=format&fit=crop&w=900&q=60' }
];

function mapProfessorToResponse(prof) {
    return {
        id: prof.id || prof.obterId?.(),
        name: prof.nome || prof.obterNome?.(),
        email: prof.email || prof.obterEmail?.(),
        instrument: prof.instrumento || prof.obterInstrumento?.(),
        price: Number(prof.preco || prof.preco_hora || prof.obterPrecoHora?.() || 0),
        bio: prof.bio || prof.biografia || '',
        image: prof.foto || prof.foto_url,
        link_video_demo: prof.link_video_demo || prof.video || null
    };
}

app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

// --- ROTAS PÚBLICAS ---

// Rota Principal: Listar Professores (Vitrine)
app.get('/api/professores', async (req, res) => {
    try {
        const professores = await professorRepo.buscarTodos();
        if (!professores || professores.length === 0) {
            return res.json(fallbackTeachers.map(mapProfessorToResponse));
        }
        res.json(professores.map(mapProfessorToResponse));
    } catch (error) {
        console.error(error);
        res.json(fallbackTeachers.map(mapProfessorToResponse));
    }
});

// Detalhes do Professor
app.get('/api/professores/:id', async (req, res) => {
    try {
        const id = Number.parseInt(req.params.id, 10);
        const teacher = await professorRepo.buscarPorId(id);
        if (teacher) {
            return res.json(mapProfessorToResponse(teacher));
        }
        const fallback = fallbackTeachers.find(t => t.id === id);
        if (fallback) return res.json(mapProfessorToResponse(fallback));
        res.status(404).json({ error: 'Professor não encontrado' });
    } catch (error) {
        res.status(500).json({ error: 'Erro interno.' });
    }
});

// --- CADASTRO COMPLETO ---
app.post('/api/register', async (req, res) => {
    // Recebe TODOS os campos do formulário
    const { name, email, password, type, bio, price, instruments, styles, videoUrl, photoUrl } = req.body;

    try {
        const salt = await bcrypt.genSalt(10);
        const senhaHash = await bcrypt.hash(password, salt);
        let userId;

        if (type === 'professor') {
            const instrumentoPrincipal = Array.isArray(instruments) && instruments.length > 0 ? instruments[0] : 'Instrumento';
            const novoProf = new Professor(null, name, email, instrumentoPrincipal, price ? Number(price) : 0, senhaHash);
            userId = await professorRepo.salvar(novoProf);
        } else {
            userId = await alunoRepo.salvar(name, email, senhaHash);
        }

        // Retorna dados para Auto-Login
        res.status(201).json({
            id: userId,
            name: name,
            email: email,
            type: type,
            token: "jwt_simulado",
            image: photoUrl,
            message: "Conta criada com sucesso!"
        });

    } catch (error) {
        console.error("Erro no cadastro:", error);
        if (error.code === '23505') { // Código de erro do Postgres para duplicidade
            return res.status(400).json({ error: "E-mail já cadastrado." });
        }
        res.status(500).json({ error: "Erro interno no servidor." });
    }
});

// --- LOGIN ---
app.post('/api/login', async (req, res) => {
    const { email, password } = req.body;

    try {
        let user = await alunoRepo.buscarPorEmail(email);
        let type = 'aluno';

        if (!user) {
            user = await professorRepo.buscarPorEmail(email);
            type = 'professor';
        }

        if (!user) {
            return res.status(400).json({ error: 'Usuário não encontrado.' });
        }

        const hashNoBanco = user.obterSenhaHash ? user.obterSenhaHash() : user.senha_hash;
        const nomeUser = user.obterNome ? user.obterNome() : user.nome;
        const idUser = user.obterId ? user.obterId() : user.id;

        if (hashNoBanco) {
            const senhaValida = await bcrypt.compare(password, hashNoBanco);
            if (!senhaValida) {
                return res.status(400).json({ error: 'Senha incorreta.' });
            }
        }

        res.json({
            id: idUser,
            name: nomeUser,
            email: email,
            type: type,
            token: "jwt_simulado"
        });

    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Erro no login.' });
    }
});

// --- PAGAMENTO ---
app.post('/api/payment', async (req, res) => {
    const { teacherId, date, time, price, studentId } = req.body; 

    try {
        // Cria objeto simples para salvar (supondo que AulaRepository aceite objeto simples)
        // Se AulaRepository exigir new Aula(), precisaria instanciar aqui.
        const objAula = {
            alunoId: studentId || 1,
            professorId: teacherId,
            dataHora: new Date(`${date}T${time}`),
            valor: price
        };

        await aulaRepo.salvar(objAula);
        res.json({ success: true, message: 'Aula contratada com sucesso!' });
    } catch (error) {
        console.error("Erro ao agendar:", error);
        res.status(500).json({ error: "Erro ao processar agendamento." });
    }
});

// Fallback
app.get(/.*/, (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(port, () => {
    console.log(`Clave Pro rodando na porta ${port}`);
});